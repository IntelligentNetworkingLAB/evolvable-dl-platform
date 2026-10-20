# PyTorch 로 돌리는 작은 MLP 와 LoRA 어댑터.
#
# 베이스 가중치는 얼리고 어댑터(A, B)만 학습해서 서버로 올린다.
# 그래서 "전체 가중치 대신 어댑터만 전송" 이라는 수치를 실제로 셀 수 있다.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_KEYS = ('A', 'B', 'W2', 'b2')


def _init(gen, shape, scale=None):
    fan_in = shape[0]
    s = scale if scale is not None else float(np.sqrt(2.0 / fan_in))
    return torch.randn(shape, generator=gen) * s


def as_tensor(x, dtype=torch.float32):
    """numpy 배열이나 텐서를 모델 입력 dtype 의 텐서로 바꾼다."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype)


class LoRAMLP(nn.Module):
    """은닉층 하나짜리 MLP. 1층 가중치에만 저랭크 어댑터를 붙인다.

    W1_eff = W1 + (alpha / rank) * A @ B
    학습·전송 대상은 A, B, 그리고 출력층(작아서 같이 보낸다)뿐이다.
    """

    def __init__(self, d_in, d_hidden, d_out, rank=8, alpha=16, seed=0, task='reg'):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out
        self.rank, self.alpha = rank, alpha
        self.task = task                      # 'reg' 또는 'clf'

        # 베이스는 전 노드가 같은 seed 로 만들어 공유한다. 학습하지 않는다.
        self.W1 = nn.Parameter(_init(gen, (d_in, d_hidden)), requires_grad=False)
        self.b1 = nn.Parameter(torch.zeros(d_hidden), requires_grad=False)

        # 어댑터. B 를 0 으로 두면 시작 시점에는 베이스와 완전히 같다.
        self.A = nn.Parameter(_init(gen, (d_in, rank), scale=0.02))
        self.B = nn.Parameter(torch.zeros(rank, d_hidden))

        self.W2 = nn.Parameter(_init(gen, (d_hidden, d_out)))
        self.b2 = nn.Parameter(torch.zeros(d_out))

    # 전송 대상 파라미터 수
    def adapter_params(self):
        return sum(getattr(self, k).numel() for k in ADAPTER_KEYS)

    def full_params(self):
        return self.adapter_params() + self.W1.numel() + self.b1.numel()

    def get_adapter(self):
        return {k: getattr(self, k).detach().clone() for k in ADAPTER_KEYS}

    def set_adapter(self, ad):
        with torch.no_grad():
            for k in ADAPTER_KEYS:
                getattr(self, k).copy_(ad[k])

    def _w1_eff(self):
        return self.W1 + (self.alpha / self.rank) * (self.A @ self.B)

    def forward(self, X):
        """회귀는 출력값, 분류는 로짓을 돌려준다."""
        h = torch.tanh(X @ self._w1_eff() + self.b1)
        return h @ self.W2 + self.b2

    @torch.no_grad()
    def predict(self, X):
        """numpy 배열을 돌려준다. 분류는 클래스 확률."""
        z = self.forward(as_tensor(X))
        if self.task == 'clf':
            z = torch.softmax(z, dim=1)
        return z.numpy().astype(np.float64)

    def _loss(self, out, y):
        if self.task == 'clf':
            return F.cross_entropy(out, y)
        return F.mse_loss(out, y.reshape(out.shape))

    def _train_loss(self, out, y):
        # 회귀 학습 목적함수는 출력 차원별 제곱오차의 합을 샘플 평균한 값이다.
        # (보고용 loss 는 전체 원소 평균 MSE)
        loss = self._loss(out, y)
        return loss * self.d_out if self.task == 'reg' else loss

    def _target(self, y):
        return as_tensor(y, torch.long if self.task == 'clf' else torch.float32)

    @torch.no_grad()
    def loss(self, X, y):
        return float(self._loss(self.forward(as_tensor(X)), self._target(y)))

    def train_epochs(self, X, y, epochs=1, lr=0.05, batch=64, gen=None):
        """로컬 학습. 어댑터와 출력층만 갱신한다."""
        gen = gen or torch.Generator().manual_seed(0)
        Xt, yt = as_tensor(X), self._target(y)
        n = Xt.shape[0]
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=lr)
        for _ in range(max(1, epochs)):
            idx = torch.randperm(n, generator=gen)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                opt.zero_grad()
                self._train_loss(self.forward(Xt[bi]), yt[bi]).backward()
                nn.utils.clip_grad_value_(params, 5.0)
                opt.step()
        return self.loss(Xt, yt)
