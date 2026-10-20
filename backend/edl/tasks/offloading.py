# 추론 오프로딩.
#
# 요청마다 엣지에서 끝낼지 클라우드로 올릴지 분류기가 정한다.
# 그 결정대로 실제 로그를 재생해서 처리율, 지연, 에너지, 품질을 잰다.
# 비교군은 학습 없이 노드 부하만 보고 자르는 고정 규칙이다.
#
# 문턱값은 응답 품질 손실 1%p 이내라는 제약을 두고
# 검증 구간에서 고른다. 품질을 지키는 선에서 엣지 처리율을 최대로 밀어붙인다.

import numpy as np

from ..datasets import offload_logs
from .base import Task, kpi

NODES = ['SBS-01', 'SBS-02', 'SBS-03', 'UAA-01', 'DRONE-01', 'VEH-01']
TIERS = ['hi', 'mid', 'lo', 'mid', 'lo', 'mid']

QUALITY_BUDGET = 1.0            # 허용 품질 손실 %p
E_EDGE, E_CLOUD = 1.0, 4.2      # 요청당 상대 에너지


class OffloadingTask(Task):
    id = 'offloading'
    label = '추론 오프로딩'
    comp = '④ 컴퓨팅 자원 관리'
    baseline_name = '부하 기준 고정 할당'
    model_name = 'LoRA-MLP (라우팅 분류)'
    lr, epochs, batch = 0.02, 1, 64

    def build(self):
        ds = offload_logs(n_nodes=len(NODES), n_per_node=1600, seed=self.seed + 2)
        X, y, meta = ds['X'], ds['y'], ds['meta']
        n_per = ds['n_per_node']

        shards = []
        va, ev = [], []
        for b, node in enumerate(NODES):
            s = b * n_per
            tr_e = s + int(n_per * 0.60)
            va_e = s + int(n_per * 0.75)
            e = s + n_per
            shards.append((node, X[s:tr_e], y[s:tr_e], TIERS[b]))
            va.append((slice(tr_e, va_e)))
            ev.append((slice(va_e, e)))

        def gather(sl):
            return {'X': np.concatenate([X[s] for s in sl]),
                    'y': np.concatenate([y[s] for s in sl]),
                    'meta': np.concatenate([meta[s] for s in sl]),
                    'node': np.concatenate([np.full(s.stop - s.start, i)
                                            for i, s in enumerate(sl)])}

        self._va, self._ev = gather(va), gather(ev)
        self._make(shards, d_hidden=24, d_out=2, task='clf',
                   ranks={'hi': 16, 'mid': 16, 'lo': 8, 'xlo': 4})

    def _drift_eval(self, node_index, shift, scale):
        for sp in (self._va, self._ev):
            m = sp['node'] == node_index
            sp['X'][m] = sp['X'][m] * scale + shift

    def _serve(self, split, to_edge):
        """라우팅 결정대로 재생. 엣지는 틀릴 수 있고 클라우드는 항상 맞다."""
        y, meta = split['y'], split['meta']
        size, load = meta[:, 0], meta[:, 2]
        lat = np.where(to_edge, 8.0 + 40.0 * load,
                       95.0 + 48.0 * np.log1p(size))
        correct = np.where(to_edge, y == 1, True)
        energy = np.where(to_edge, E_EDGE, E_CLOUD)
        return {
            'edge': 100.0 * to_edge.mean(),
            'lat': float(lat.mean()),
            'quality': 100.0 * correct.mean(),
            'energy': 100.0 * energy.sum() / (E_CLOUD * len(y)),
        }

    def _fixed(self, split):
        return self._serve(split, split['meta'][:, 2] < 0.30)

    def _pick_tau(self, model):
        """검증 구간에서 품질 예산을 지키는 가장 공격적인 문턱값."""
        floor = self._fixed(self._va)['quality'] - QUALITY_BUDGET
        p = model.predict(self._va['X'])[:, 1]
        best, best_edge = 0.95, -1.0
        for tau in np.arange(0.30, 0.96, 0.025):
            r = self._serve(self._va, p >= tau)
            if r['quality'] >= floor and r['edge'] > best_edge:
                best, best_edge = float(tau), r['edge']
        return best

    def evaluate(self):
        model = self.server.global_model(task='clf')
        tau = self._pick_tau(model)
        p = model.predict(self._ev['X'])[:, 1]
        ours = self._serve(self._ev, p >= tau)

        if self.baseline_cache is None:
            self.baseline_cache = self._fixed(self._ev)
        base = self.baseline_cache

        loss = float(np.mean([model.loss(c.X, c.y) for c in self.clients]))
        return {
            'loss': loss,
            'tau': round(tau, 3),
            'kpis': [
                kpi('edge', '엣지 처리율', '%', ours['edge'], base['edge'], 'up'),
                kpi('lat', '평균 추론 지연', 'ms', ours['lat'], base['lat'], 'down', dec=0),
                kpi('energy', '에너지 소비', '%', ours['energy'], base['energy'], 'down'),
                kpi('quality', '응답 품질', '%', ours['quality'], base['quality'], 'up'),
            ],
        }
