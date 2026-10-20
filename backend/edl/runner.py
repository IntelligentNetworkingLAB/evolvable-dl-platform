# 실행 하나를 붙잡고 라운드를 진행시킨다. 이벤트도 여기서 건다.

import threading
import time
import uuid

from .tasks import REGISTRY

MAX_RUNS = 8


class Run:
    def __init__(self, task_id, seed=0, rounds=24):
        if task_id not in REGISTRY:
            raise KeyError(task_id)
        self.id = uuid.uuid4().hex[:12]
        self.task_id = task_id
        self.rounds = rounds
        self.created = time.time()
        self.task = REGISTRY[task_id](seed=seed)
        self.history = []
        self.events = []
        self.pending_redistill = 0
        self.lock = threading.Lock()
        base = self.task.evaluate()
        self.history.append({'round': 0, 'loss': base['loss'],
                             'kpis': base['kpis'], 'selected': [],
                             'client_loss': {}, 'comm': self.task.comm(0)})

    def nodes(self):
        out = []
        for c in self.task.clients:
            out.append({
                'id': c.id, 'tier': c.tier, 'rank': c.rank,
                'samples': int(c.n),
                'adapter_bytes': int(c.uplink_bytes()),
                'full_bytes': int(c.full_bytes()),
                'data_bytes': int(c.data_bytes()),
                'drifted': bool(c.drifted),
            })
        return out

    def meta(self):
        t = self.task
        return {'run_id': self.id, 'task': t.id, 'label': t.label,
                'comp': t.comp, 'baseline': t.baseline_name,
                'model': t.model_name, 'rounds': self.rounds,
                'lr': t.lr, 'epochs': t.epochs, 'batch': t.batch,
                'nodes': self.nodes()}

    def step(self):
        with self.lock:
            if self.task.round >= self.rounds:
                return self.history[-1]
            if self.pending_redistill > 0:
                self.pending_redistill -= 1
                if self.pending_redistill == 0:
                    node = self.task.redistill()
                    self.events.append({'round': self.task.round,
                                        'kind': 'redistill', 'node': node})
            info = self.task.step()
            info['nodes'] = self.nodes()
            self.history.append(info)
            return info

    def event(self, kind):
        """E1 분포 급변, E2 노드 합류, E3 재증류."""
        with self.lock:
            if kind == 'drift':
                node = self.task.inject_drift('SBS-02', strength=1.0)
            elif kind == 'join':
                node = self.task.join_client('DRONE-03', tier='xlo', rank=4)
            elif kind == 'cascade':
                node = self.task.inject_drift('SBS-03', strength=0.7)
                self.pending_redistill = 2      # 두 라운드 뒤 자동 재증류
            else:
                raise ValueError(kind)
            self.events.append({'round': self.task.round, 'kind': kind,
                                'node': node})
            return {'kind': kind, 'node': node, 'round': self.task.round}


class Runner:
    def __init__(self):
        self.runs = {}
        self.lock = threading.Lock()

    def create(self, task_id, seed=0, rounds=24):
        run = Run(task_id, seed=seed, rounds=rounds)
        with self.lock:
            if len(self.runs) >= MAX_RUNS:
                oldest = min(self.runs.values(), key=lambda r: r.created)
                self.runs.pop(oldest.id, None)
            self.runs[run.id] = run
        return run

    def get(self, run_id):
        return self.runs.get(run_id)

    def drop(self, run_id):
        with self.lock:
            return self.runs.pop(run_id, None) is not None


def task_catalog():
    out = []
    for tid, cls in REGISTRY.items():
        out.append({'id': tid, 'label': cls.label, 'comp': cls.comp,
                    'baseline': cls.baseline_name, 'model': cls.model_name})
    return out
