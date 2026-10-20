# FedAvg 와 클라이언트. 데이터는 클라이언트 밖으로 나가지 않고
# 어댑터만 서버로 올라간다.

import numpy as np
import torch

from .model import LoRAMLP

FLOAT_BYTES = 4

# 하드웨어 등급별 경량화. 업링크 전에 실제로 적용한다.
QUANT_BITS = {'hi': 16, 'mid': 8, 'lo': 4, 'xlo': 4}
PRUNE_RATE = {'hi': 0.0, 'mid': 0.18, 'lo': 0.38, 'xlo': 0.62}


def quantize(a, bits):
    """대칭 per-tensor 양자화. 값을 실제로 뭉갠다."""
    if bits >= 32:
        return a
    m = a.abs().max() if a.numel() else a.new_zeros(())
    if m == 0:
        return a
    step = m / (2 ** (bits - 1) - 1)
    return torch.round(a / step) * step


def prune(a, rate):
    """크기 작은 원소부터 0 으로 만든다."""
    if rate <= 0 or a.numel() == 0:
        return a
    k = int(a.numel() * rate)
    if k <= 0:
        return a
    mag = a.abs()
    thr = torch.kthvalue(mag.flatten(), k).values
    return torch.where(mag <= thr, torch.zeros_like(a), a)


def fedavg(adapters, sizes):
    """샘플 수로 가중 평균."""
    if not adapters:
        raise ValueError('빈 어댑터 목록')
    w = torch.as_tensor(sizes, dtype=torch.float64)
    w = (w / w.sum()).tolist()
    out = {}
    for k in adapters[0]:
        out[k] = sum(a[k] * wi for a, wi in zip(adapters, w))
    return out


class Client:
    """엣지 노드 하나. 자기 몫의 데이터만 들고 있다."""

    def __init__(self, node_id, X, y, tier='mid', rank=8, seed=0,
                 d_hidden=32, d_out=1, task='reg'):
        self.id = node_id
        self.X, self.y = X, y
        self.tier = tier
        self.rank = rank
        self.rng = np.random.default_rng(seed)          # 드리프트용
        self.gen = torch.Generator().manual_seed(seed)  # 미니배치 순서용
        self.model = LoRAMLP(X.shape[1], d_hidden, d_out,
                             rank=rank, seed=0, task=task)
        self.last_loss = None
        self.drifted = False

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def bits(self):
        return QUANT_BITS.get(self.tier, 8)

    @property
    def prune_rate(self):
        return PRUNE_RATE.get(self.tier, 0.0)

    def uplink_bytes(self):
        # 양자화 비트폭이 그대로 전송량이 된다
        return int(self.model.adapter_params() * self.bits / 8)

    def full_bytes(self):
        return self.model.full_params() * FLOAT_BYTES

    def data_bytes(self):
        return int(self.X.nbytes + self.y.nbytes)

    def train(self, global_adapter, epochs=1, lr=0.05, batch=64):
        self.model.set_adapter(self._fit_rank(global_adapter))
        self.last_loss = self.model.train_epochs(
            self.X, self.y, epochs=epochs, lr=lr, batch=batch, gen=self.gen)
        # 올리기 전에 등급대로 프루닝하고 양자화한다. 저사양 노드는
        # 그만큼 손실이 생기고 그게 집계 결과에 실제로 반영된다.
        ad = self.model.get_adapter()
        out = {}
        for k, v in ad.items():
            v = prune(v, self.prune_rate) if k in ('A', 'B') else v
            out[k] = quantize(v, self.bits)
        return out, self.n, self.last_loss

    def _fit_rank(self, ad):
        """저사양 노드는 rank 가 작다. 글로벌 어댑터를 잘라서 받는다."""
        r = self.model.rank
        if ad['A'].shape[1] == r:
            return ad
        return {'A': ad['A'][:, :r].clone(), 'B': ad['B'][:r, :].clone(),
                'W2': ad['W2'].clone(), 'b2': ad['b2'].clone()}

    def apply_drift(self, strength=1.0):
        """입력 분포에 이동을 주어 드리프트를 만든다."""
        k = self.X.shape[1]
        shift = self.rng.normal(0.0, 0.9 * strength, size=k)
        scale = 1.0 + self.rng.normal(0.0, 0.35 * strength, size=k)
        self.X = self.X * scale + shift
        self.drifted = True
        return shift, scale


class Server:
    """글로벌 어댑터를 들고 라운드를 돌린다."""

    def __init__(self, d_in, d_hidden, d_out, rank=8, task='reg', seed=0):
        self.template = LoRAMLP(d_in, d_hidden, d_out, rank=rank,
                                seed=0, task=task)
        self.adapter = self.template.get_adapter()
        self.rng = np.random.default_rng(seed)
        self.rank = rank

    def round(self, clients, frac=0.75, epochs=1, lr=0.05, batch=64):
        n = max(2, int(round(len(clients) * frac)))
        picked = list(self.rng.choice(clients, size=min(n, len(clients)),
                                      replace=False))
        ups, sizes, losses, uplink = [], [], {}, 0
        for c in picked:
            ad, m, loss = c.train(self.adapter, epochs=epochs, lr=lr, batch=batch)
            # rank 가 작은 노드는 0 으로 채워 글로벌 rank 에 맞춘다.
            if ad['A'].shape[1] != self.rank:
                A = torch.zeros(ad['A'].shape[0], self.rank)
                B = torch.zeros(self.rank, ad['B'].shape[1])
                A[:, :ad['A'].shape[1]] = ad['A']
                B[:ad['B'].shape[0], :] = ad['B']
                ad = {'A': A, 'B': B, 'W2': ad['W2'], 'b2': ad['b2']}
            ups.append(ad)
            sizes.append(m)
            losses[c.id] = loss
            uplink += c.uplink_bytes()

        self.adapter = fedavg(ups, sizes)
        return {
            'selected': [c.id for c in picked],
            'client_loss': losses,
            'uplink_bytes': uplink,
        }

    def global_model(self, task='reg'):
        m = LoRAMLP(self.template.d_in, self.template.d_hidden,
                    self.template.d_out, rank=self.rank, seed=0, task=task)
        m.set_adapter(self.adapter)
        return m
