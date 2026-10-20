import numpy as np
import pytest

from edl.datasets import (caching_features, channel_traces, content_requests,
                          offload_logs)
from edl.runner import Runner
from edl.tasks import REGISTRY

ALL = sorted(REGISTRY)


def test_request_trace_is_reproducible():
    a = content_requests(n_bs=2, n_content=20, n_slots=30, seed=7)
    b = content_requests(n_bs=2, n_content=20, n_slots=30, seed=7)
    assert np.array_equal(a['counts'], b['counts'])


def test_request_trace_is_non_iid_across_cells():
    tr = content_requests(n_bs=4, n_content=40, n_slots=60, seed=1)
    tot = tr['counts'].sum(axis=1)
    top = [np.argmax(tot[b]) for b in range(4)]
    assert len(set(top)) > 1


def test_caching_features_align_with_counts():
    tr = content_requests(n_bs=2, n_content=15, n_slots=40, seed=2)
    ds = caching_features(tr, lags=4, warmup=8)
    assert ds['X'].shape[0] == ds['y'].shape[0] == len(ds['t'])
    assert ds['X'].shape[1] == 8


def test_offload_labels_are_binary():
    ds = offload_logs(n_nodes=2, n_per_node=200, seed=3)
    assert set(np.unique(ds['y']).tolist()) <= {0, 1}


def test_channel_is_time_correlated():
    # 심볼 간 상관이 없으면 다음 심볼을 예측할 근거가 사라진다
    tr = channel_traces(n_bs=1, n_sub=16, n_sym=400, seed=4)
    h = tr['H'][0]
    c = np.abs(np.vdot(h[:-1].ravel(), h[1:].ravel())) / np.sum(np.abs(h) ** 2)
    assert c > 0.5


@pytest.mark.parametrize('tid', ALL)
def test_task_builds_with_clients(tid):
    t = REGISTRY[tid](seed=0)
    assert len(t.clients) >= 3
    assert all(c.n > 0 for c in t.clients)


@pytest.mark.parametrize('tid', ALL)
def test_round_reports_expected_shape(tid):
    t = REGISTRY[tid](seed=0)
    info = t.step()
    assert info['round'] == 1
    assert info['kpis'] and all('value' in k for k in info['kpis'])
    assert info['comm']['adapter_bytes_this_round'] > 0
    assert info['selected']


@pytest.mark.parametrize('tid', ALL)
def test_training_improves_primary_kpi_over_baseline(tid):
    """24라운드를 돌면 주 지표가 학습 없는 기준선을 넘어야 한다."""
    t = REGISTRY[tid](seed=0)
    for _ in range(24):
        info = t.step()
    k = info['kpis'][0]
    if k['dir'] == 'up':
        assert k['value'] > k['baseline']
    else:
        assert k['value'] < k['baseline']


@pytest.mark.parametrize('tid', ALL)
def test_uplink_is_far_smaller_than_local_data(tid):
    # 채널 태스크는 출력 차원이 커서 어댑터 비중이 상대적으로 높다.
    # 그래도 원본 데이터의 2% 아래여야 한다.
    t = REGISTRY[tid](seed=0)
    info = t.step()
    c = info['comm']
    assert c['adapter_bytes_this_round'] < c['data_bytes'] / 50
    assert c['saving_vs_data_pct'] > 98.0


def test_drift_degrades_then_recovers():
    t = REGISTRY['caching'](seed=0)
    for _ in range(10):
        info = t.step()
    before = info['kpis'][0]['value']
    t.inject_drift('SBS-02', strength=1.0)
    after = t.step()['kpis'][0]['value']
    assert after < before
    for _ in range(6):
        info = t.step()
    assert info['kpis'][0]['value'] > after


def test_joining_node_gets_smaller_rank():
    t = REGISTRY['caching'](seed=0)
    n0 = len(t.clients)
    t.join_client('DRONE-03', tier='xlo', rank=4)
    assert len(t.clients) == n0 + 1
    joined = t.clients[-1]
    assert joined.rank == 4
    assert joined.n < t.clients[0].n
    t.step()


def test_runner_round_trip():
    r = Runner()
    run = r.create('caching', seed=0)
    assert run.history[0]['round'] == 0
    out = run.step()
    assert out['round'] == 1
    assert r.get(run.id) is run
    ev = run.event('drift')
    assert ev['node']
    assert r.drop(run.id)
    assert r.get(run.id) is None


def test_runner_rejects_unknown_task():
    with pytest.raises(KeyError):
        Runner().create('nope')
