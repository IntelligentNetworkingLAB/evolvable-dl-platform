# 채널 추정.
#
# 파일럿 관측으로 다음 심볼의 전 서브캐리어 채널을 추정한다.
# 비교군은 파일럿 위치에서 LS 로 뽑고 사이를 선형 보간하는 전통 방식.
# NMSE 와 스펙트럼 효율 둘 다 추정 오차에서 직접 나온 값이다.

import numpy as np

from ..datasets import channel_features, channel_traces
from .base import Task, kpi

NODES = ['MBS-01', 'SBS-01', 'SBS-02', 'SBS-03', 'UAA-01', 'SAT-01']
TIERS = ['hi', 'hi', 'mid', 'lo', 'mid', 'xlo']


def _nmse(est, ref):
    return float(np.sum(np.abs(est - ref) ** 2) / np.sum(np.abs(ref) ** 2))


def _spectral_eff(H, err_var, snr_lin):
    """추정 오차를 간섭으로 본 유효 SINR 기준 평균 용량."""
    g = np.abs(H) ** 2
    sinr = snr_lin * g / (1.0 + snr_lin * err_var)
    return float(np.mean(np.log2(1.0 + sinr)))


class ChannelTask(Task):
    id = 'channel'
    label = '채널추정 · 자원할당'
    comp = '⑤ 네트워크 자원 관리'
    baseline_name = 'LS 추정 + 선형 보간'
    model_name = 'LoRA-MLP (채널 추정)'
    lr, epochs, batch = 0.03, 2, 64

    def build(self):
        tr = channel_traces(n_bs=len(NODES), n_sub=32, n_sym=520,
                            pilot_step=4, snr_db=12.0, seed=self.seed + 3)
        ds = channel_features(tr)
        self.n_sub = ds['n_sub']
        self.snr_lin = 10 ** (tr['snr_db'] / 10.0)
        self.pilots = tr['pilots']
        self._raw = tr
        self.scale = ds['scale']

        X, y, bs = ds['X'], ds['y'], ds['bs']
        n_sym = tr['H'].shape[1] - 1
        cut = int(n_sym * 0.75)

        shards, ev_X, ev_y, ev_i = [], [], [], []
        for b, node in enumerate(NODES):
            m = bs == b
            Xb, yb = X[m], y[m]
            shards.append((node, Xb[:cut], yb[:cut], TIERS[b]))
            ev_X.append(Xb[cut:]); ev_y.append(yb[cut:])
            ev_i.append(np.full(len(Xb) - cut, b))
        self._ev = {'X': np.concatenate(ev_X), 'y': np.concatenate(ev_y),
                    'bs': np.concatenate(ev_i), 'cut': cut}
        self._make(shards, d_hidden=64, d_out=2 * self.n_sub, task='reg',
                   ranks={'hi': 16, 'mid': 16, 'lo': 8, 'xlo': 4})

    def _drift_eval(self, node_index, shift, scale):
        m = self._ev['bs'] == node_index
        self._ev['X'][m] = self._ev['X'][m] * scale + shift

    def _to_complex(self, arr):
        return arr[:, :self.n_sub] + 1j * arr[:, self.n_sub:]

    def _ls_baseline(self):
        """파일럿에서 LS, 나머지는 선형 보간. 학습 없음."""
        if self.baseline_cache is not None:
            return self.baseline_cache
        H, Y, p = self._raw['H'], self._raw['Y'], self.pilots
        cut = self._ev['cut']
        est, ref = [], []
        k = np.arange(self.n_sub)
        for b in range(len(NODES)):
            for t in range(cut, H.shape[1] - 1):
                obs = Y[b, t]                       # 현재 심볼 파일럿 관측
                re = np.interp(k, p, obs.real)
                im = np.interp(k, p, obs.imag)
                est.append(re + 1j * im)
                ref.append(H[b, t + 1])             # 다음 심볼이 정답
        est, ref = np.array(est), np.array(ref)
        nm = _nmse(est, ref)
        se = _spectral_eff(ref, nm, self.snr_lin)
        self.baseline_cache = (nm, se)
        return self.baseline_cache

    def evaluate(self):
        model = self.server.global_model(task='reg')
        pred = model.predict(self._ev['X'])
        est = self._to_complex(pred)
        ref = self._to_complex(self._ev['y'])

        nm = _nmse(est, ref)
        se = _spectral_eff(ref, nm, self.snr_lin)
        b_nm, b_se = self._ls_baseline()

        loss = float(np.mean([model.loss(c.X[:3000], c.y[:3000])
                              for c in self.clients]))
        return {
            'loss': loss,
            'kpis': [
                kpi('nmse', '채널추정 NMSE', 'dB', 10 * np.log10(nm),
                    10 * np.log10(b_nm), 'down'),
                kpi('se', '하향 스펙트럼 효율', 'b/s/Hz', se, b_se, 'up', dec=2),
            ],
        }
