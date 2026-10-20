# numpy 만으로 돌리는 작은 MLP 와 LoRA 어댑터.
#
# 베이스 가중치는 얼지고 어댑터(A, B)만 학습해서 서버로 올린다.
# 그래서 "전체 가중치 대신 어댑터만 전송" 이라는 수치를 실제로 셀 수 있다.

import numpy as np


def _init(rng, shape, scale=None):
    fan_in = shape[0]
    s = scale if scale is not None else np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, s, size=shape)


class LoRAMLP:
    """은닉층 하나짜리 MLP. 1층 가중치에만 저랭크 어댑터를 붙인다.

    W1_eff = W1 + (alpha / rank) * A @ B
    학습·전송 대상은 A, B, 그리고 출력층(작아서 같이 보낸다)뿐이다.
    """

    def __init__(self, d_in, d_hidden, d_out, rank=8, alpha=16, seed=0, task='reg'):
        rng = np.random.default_rng(seed)
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out
        self.rank, self.alpha = rank, alpha
        self.task = task                      # 'reg' 또는 'clf'

        # 베이스는 전 노드가 같은 seed 로 만들어 공유한다. 학습하지 않는다.
        self.W1 = _init(rng, (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)

        # 어댑터. B 를 0 으로 두면 시작 시점에는 베이스와 완전히 같다.
        self.A = _init(rng, (d_in, rank), scale=0.02)
        self.B = np.zeros((rank, d_hidden))

        self.W2 = _init(rng, (d_hidden, d_out))
        self.b2 = np.zeros(d_out)

    # 전송 대상 파라미터 수
    def adapter_params(self):
        return self.A.size + self.B.size + self.W2.size + self.b2.size

    def full_params(self):
        return (self.W1.size + self.b1.size + self.A.size + self.B.size
                + self.W2.size + self.b2.size)

    def get_adapter(self):
        return {'A': self.A.copy(), 'B': self.B.copy(),
                'W2': self.W2.copy(), 'b2': self.b2.copy()}

    def set_adapter(self, ad):
        self.A = ad['A'].copy()
        self.B = ad['B'].copy()
        self.W2 = ad['W2'].copy()
        self.b2 = ad['b2'].copy()

    def _w1_eff(self):
        return self.W1 + (self.alpha / self.rank) * (self.A @ self.B)

    def forward(self, X):
        z1 = X @ self._w1_eff() + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        if self.task == 'clf':
            z2 = z2 - z2.max(axis=1, keepdims=True)
            e = np.exp(z2)
            out = e / e.sum(axis=1, keepdims=True)
        else:
            out = z2
        return out, h, z1

    def predict(self, X):
        return self.forward(X)[0]

    def loss(self, X, y):
        out, _, _ = self.forward(X)
        if self.task == 'clf':
            n = X.shape[0]
            p = np.clip(out[np.arange(n), y], 1e-12, 1.0)
            return float(-np.log(p).mean())
        return float(((out - y) ** 2).mean())

    def train_epochs(self, X, y, epochs=1, lr=0.05, batch=64, rng=None):
        """로컬 학습. 어댑터와 출력층만 갱신한다."""
        rng = rng or np.random.default_rng(0)
        n = X.shape[0]
        scale = self.alpha / self.rank
        last = None
        for _ in range(max(1, epochs)):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                xb, yb = X[bi], y[bi]
                m = xb.shape[0]
                out, h, z1 = self.forward(xb)

                if self.task == 'clf':
                    g2 = out.copy()
                    g2[np.arange(m), yb] -= 1.0
                    g2 /= m
                else:
                    g2 = 2.0 * (out - yb) / m

                gW2 = h.T @ g2
                gb2 = g2.sum(axis=0)

                gh = g2 @ self.W2.T
                gz1 = gh * (1.0 - h ** 2)

                # W1_eff = W1 + scale * A @ B 이므로 A, B 기울기는 다음과 같다.
                gW1eff = xb.T @ gz1
                gA = scale * (gW1eff @ self.B.T)
                gB = scale * (self.A.T @ gW1eff)

                for p, g in ((self.W2, gW2), (self.b2, gb2),
                             (self.A, gA), (self.B, gB)):
                    np.clip(g, -5.0, 5.0, out=g)
                    p -= lr * g
            last = self.loss(X, y)
        return last
