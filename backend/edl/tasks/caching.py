# 콘텐츠 캐싱.
#
# 다음 슬롯 요청량을 예측해서 캐시를 채우고, 실제 요청 스트림에 재생해
# 히트율을 잰다. 비교군은 같은 스트림에 LRU 를 돌린 결과다.
# 히트율, 백홀, 재사용률은 전부 재생 결과에서 나온 값이고
# 지연만 히트/미스에 고정 상수를 곱해 환산한 값이다.

import numpy as np

from ..datasets import caching_features, content_requests
from .base import Task, kpi

LAT_HIT = 3.1          # ms, 엣지 캐시에서 바로 나갈 때
LAT_MISS = 24.0        # ms, 백홀까지 갔다 올 때
NODES = ['SBS-01', 'SBS-02', 'SBS-03', 'UAA-01', 'DRONE-01', 'VEH-01']
TIERS = ['hi', 'mid', 'lo', 'mid', 'lo', 'mid']


class CachingTask(Task):
    id = 'caching'
    label = '콘텐츠·모델 캐싱'
    comp = '③ 자율·연계통합 데이터 분석'
    baseline_name = 'LRU 캐시 정책'
    model_name = 'LoRA-MLP (인기도 회귀)'

    def build(self):
        self.n_bs = len(NODES)
        tr = content_requests(n_bs=self.n_bs, n_content=80, n_slots=220,
                              seed=self.seed + 1)
        ds = caching_features(tr, lags=4, warmup=8)
        self._trace = tr
        self._norm = (ds['mu'], ds['sd'])
        self._split = None
        self.counts = ds['counts']
        self.n_content = self.counts.shape[2]
        self.cache_k = max(6, self.n_content // 5)

        t = ds['t']
        split = int(np.quantile(t, 0.78))
        self.eval_slots = sorted(set(t[t >= split].tolist()))[:-1]

        self._split = split
        shards = []
        tr_mask = t < split
        for b in range(self.n_bs):
            m = tr_mask & (ds['bs'] == b)
            shards.append((NODES[b], ds['X'][m], ds['y'][m], TIERS[b]))
        self._eval = {'X': ds['X'], 'y': ds['y'], 't': t, 'bs': ds['bs']}
        self._make(shards, d_hidden=32, d_out=1, task='reg',
                   ranks={'hi': 16, 'mid': 16, 'lo': 8, 'xlo': 4})

    def _drift_eval(self, node_index, shift, scale):
        # 특징 공간을 흔드는 방식은 상위 K개 순위에 잘 안 먹는다.
        # 캐싱에서 분포가 바뀐다는 건 그 셀의 콘텐츠 선호가 바뀐다는 뜻이므로
        # 요청량 자체를 바꾸고 특징을 다시 만든다.
        rng = np.random.default_rng(self.seed + 500 + node_index)
        n_c = self.n_content
        hot = rng.choice(n_c, size=max(4, n_c // 6), replace=False)
        w = np.full(n_c, 0.55)
        w[hot] = 5.5
        c = self.counts[node_index].astype(np.float64) * w
        self.counts[node_index] = np.round(c).astype(np.int32)
        self._rebuild(node_index)

    def _rebuild(self, node_index):
        """바뀐 요청량으로 그 셀의 학습/평가 데이터를 다시 만든다.
        정규화 상수는 원래 것을 그대로 써서 다른 셀은 건드리지 않는다."""
        self._trace['counts'] = self.counts
        ds = caching_features(self._trace, lags=4, warmup=8, norm=self._norm)
        t, bs = ds['t'], ds['bs']
        m_all = bs == node_index
        m_tr = m_all & (t < self._split)
        m_ev = self._eval['bs'] == node_index

        self.clients[node_index].X = ds['X'][m_tr]
        self.clients[node_index].y = ds['y'][m_tr]
        self._eval['X'][m_ev] = ds['X'][m_all]
        self._eval['y'][m_ev] = ds['y'][m_all]
        self.baseline_cache = None

    def _replay(self, model):
        """예측 상위 K 개를 캐시에 올리고 다음 슬롯 요청으로 채점."""
        X, t, bs = self._eval['X'], self._eval['t'], self._eval['bs']
        hits = tot = 0
        carry = []
        prev = {}
        for slot in self.eval_slots:
            for b in range(self.n_bs):
                m = (t == slot) & (bs == b)
                if not m.any():
                    continue
                pred = model.predict(X[m]).ravel()
                top = np.argpartition(-pred, self.cache_k - 1)[:self.cache_k]
                actual = self.counts[b, slot + 1]
                hits += int(actual[top].sum())
                tot += int(actual.sum())
                if b in prev:
                    carry.append(len(set(top.tolist()) & prev[b]) / self.cache_k)
                prev[b] = set(top.tolist())
        hit = 100.0 * hits / max(1, tot)
        return hit, 100.0 * float(np.mean(carry)) if carry else 0.0

    def _lru(self):
        """같은 스트림에 LRU. 슬롯 단위 최근 사용 기준."""
        if self.baseline_cache is not None:
            return self.baseline_cache
        hits = tot = 0
        carry = []
        prev = {}
        recency = np.full((self.n_bs, self.n_content), -1, dtype=np.int64)
        first = self.eval_slots[0]
        for slot in range(first):
            for b in range(self.n_bs):
                recency[b, self.counts[b, slot] > 0] = slot
        for slot in self.eval_slots:
            for b in range(self.n_bs):
                recency[b, self.counts[b, slot] > 0] = slot
                top = np.argpartition(-recency[b], self.cache_k - 1)[:self.cache_k]
                actual = self.counts[b, slot + 1]
                hits += int(actual[top].sum())
                tot += int(actual.sum())
                if b in prev:
                    carry.append(len(set(top.tolist()) & prev[b]) / self.cache_k)
                prev[b] = set(top.tolist())
        self.baseline_cache = (100.0 * hits / max(1, tot),
                               100.0 * float(np.mean(carry)) if carry else 0.0)
        return self.baseline_cache

    def evaluate(self):
        model = self.server.global_model(task='reg')
        hit, reuse = self._replay(model)
        b_hit, b_reuse = self._lru()

        def lat(h):
            return (h / 100.0) * LAT_HIT + (1 - h / 100.0) * LAT_MISS

        loss = float(np.mean([model.loss(c.X[:2000], c.y[:2000])
                              for c in self.clients]))
        return {
            'loss': loss,
            'kpis': [
                kpi('hit', '캐시 히트율', '%', hit, b_hit, 'up'),
                kpi('bh', '백홀 전송량', '%', 100 - hit, 100 - b_hit, 'down'),
                kpi('reuse', '캐시 재사용률', '%', reuse, b_reuse, 'up'),
                kpi('lat', '평균 응답 지연', 'ms', lat(hit), lat(b_hit), 'down'),
            ],
        }
