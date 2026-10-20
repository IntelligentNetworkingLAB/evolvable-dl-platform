# 태스크 공통 뼈대.
#
# 각 태스크는 클라이언트를 만들고, 라운드를 돌리고, 글로벌 모델로
# 도메인 KPI 를 실제로 계산한다. baseline 은 학습을 안 쓰는 기존 방식이다.

import numpy as np
import torch

from ..fed import Client, Server


class Task:
    id = ''
    label = ''
    comp = ''
    baseline_name = ''
    model_name = ''
    lr = 0.012
    epochs = 1
    batch = 128

    def __init__(self, seed=0, rounds=24, lr=None, epochs=None, batch=None):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rounds = rounds
        self.lr = lr or type(self).lr
        self.epochs = epochs or type(self).epochs
        self.batch = batch or type(self).batch
        self.round = 0
        self.clients = []
        self.server = None
        self.baseline_cache = None
        self.build()

    def build(self):
        raise NotImplementedError

    def evaluate(self):
        """{'kpis': [...], 'loss': float} 를 돌려준다."""
        raise NotImplementedError

    def _make(self, shards, d_hidden, d_out, task, ranks=None):
        d_in = shards[0][1].shape[1]
        self.server = Server(d_in, d_hidden, d_out, rank=16,
                             task=task, seed=self.seed)
        self.clients = []
        for i, (node_id, X, y, tier) in enumerate(shards):
            r = (ranks or {}).get(tier, 16)
            self.clients.append(Client(node_id, X, y, tier=tier, rank=r,
                                       seed=self.seed + 17 * i,
                                       d_hidden=d_hidden, d_out=d_out,
                                       task=task))

    def step(self, frac=0.75):
        info = self.server.round(self.clients, frac=frac, epochs=self.epochs,
                                 lr=self.lr, batch=self.batch)
        self.round += 1
        ev = self.evaluate()
        info.update(ev)
        info['round'] = self.round
        info['comm'] = self.comm(info['uplink_bytes'])
        return info

    def comm(self, uplink_bytes):
        data = sum(c.data_bytes() for c in self.clients)
        full = sum(c.full_bytes() for c in self.clients)
        return {
            'data_bytes': data,
            'full_bytes_per_round': full,
            'adapter_bytes_this_round': uplink_bytes,
            # 데이터를 그대로 올렸을 때와 비교
            'saving_vs_data_pct': round(100.0 * (1.0 - uplink_bytes / max(1, data)), 4),
            # 전체 가중치를 매 라운드 올렸을 때 대비
            'saving_vs_full_pct': round(100.0 * (1.0 - uplink_bytes / max(1, full)), 3),
        }

    # 진화 이벤트
    def inject_drift(self, node_id=None, strength=1.0):
        """한 노드만 흔들면 6분의 1로 희석돼 글로벌 지표에 거의 안 보인다.
        실제로 경기장 이벤트 같은 상황은 인근 셀이 같이 영향을 받으므로
        지정 노드와 이웃 하나를 함께 흔든다."""
        c = self._find(node_id) or self.clients[min(1, len(self.clients) - 1)]
        self._drift_one(c, strength)
        i = self.clients.index(c)
        for j in (i + 1, i + 2):
            if j < len(self.clients):
                self._drift_one(self.clients[j], strength * 0.75)
        self.baseline_cache = None
        return c.id

    def _drift_one(self, c, strength):
        shift, scale = c.apply_drift(strength)
        # 분포가 바뀌었으면 앞으로 들어올 데이터도 바뀐다. 평가 구간에도 적용.
        idx = [i for i, x in enumerate(self.clients) if x.id == c.id]
        if idx:
            self._drift_eval(idx[0], shift, scale)

    def _drift_eval(self, node_index, shift, scale):
        """태스크마다 평가 세트 구조가 달라서 각자 구현한다."""

    def join_client(self, node_id='DRONE-03', tier='xlo', frac=0.25, rank=4):
        """저사양 노드가 중간에 합류한다. 데이터도 적고 rank 도 낮다."""
        src = self.clients[-1]
        n = max(64, int(src.n * frac))
        idx = np.random.default_rng(self.seed + 99).choice(src.n, n, replace=False)
        c = Client(node_id, src.X[idx].copy(), src.y[idx].copy(), tier=tier,
                   rank=rank, seed=self.seed + 123,
                   d_hidden=src.model.d_hidden, d_out=src.model.d_out,
                   task=src.model.task)
        self.clients.append(c)
        return c.id

    def redistill(self, node_id=None):
        """글로벌 어댑터를 그대로 덮어써서 열화된 로컬 모델을 되돌린다."""
        c = self._find(node_id) or self.clients[-1]
        c.model.set_adapter(c._fit_rank(self.server.adapter))
        c.drifted = False
        return c.id

    def _find(self, node_id):
        for c in self.clients:
            if c.id == node_id:
                return c
        return None


def kpi(kid, label, unit, value, baseline, direction, dec=1):
    return {'id': kid, 'label': label, 'unit': unit,
            'value': round(float(value), dec + 2),
            'baseline': None if baseline is None else round(float(baseline), dec + 2),
            'dir': direction, 'dec': dec}
