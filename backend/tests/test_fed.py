import numpy as np
import pytest

from edl.fed import Client, Server, fedavg, prune, quantize
from edl.model import LoRAMLP


def test_fedavg_weights_by_sample_count():
    a = {'A': np.zeros((2, 2)), 'B': np.zeros((2, 2)),
         'W2': np.zeros((2, 1)), 'b2': np.zeros(1)}
    b = {k: np.ones_like(v) for k, v in a.items()}
    out = fedavg([a, b], [1, 3])
    assert np.allclose(out['A'], 0.75)


def test_fedavg_rejects_empty():
    with pytest.raises(ValueError):
        fedavg([], [])


def test_quantize_reduces_distinct_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    q = quantize(a, 4)
    assert len(np.unique(q)) <= 16
    assert np.abs(q).max() <= np.abs(a).max() + 1e-9


def test_quantize_passthrough_for_float32():
    a = np.array([1.0, -2.0])
    assert np.allclose(quantize(a, 32), a)


def test_prune_zeroes_requested_fraction():
    a = np.arange(1.0, 101.0)
    p = prune(a, 0.30)
    assert (p == 0).sum() == 30
    assert p[-1] == 100.0


def test_adapter_is_smaller_than_full_model():
    m = LoRAMLP(8, 32, 1, rank=8)
    assert m.adapter_params() < m.full_params()


def test_lora_starts_equal_to_base():
    # B 가 0 이므로 학습 전에는 베이스 가중치와 같아야 한다
    m = LoRAMLP(6, 16, 1, rank=4)
    x = np.random.default_rng(0).normal(size=(5, 6))
    base = np.tanh(x @ m.W1 + m.b1) @ m.W2 + m.b2
    assert np.allclose(m.predict(x), base)


def test_local_training_reduces_loss():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    y = (X[:, :1] * 2.0 - X[:, 1:2])
    c = Client('N1', X, y, tier='hi', rank=8, d_hidden=24, d_out=1, task='reg')
    before = c.model.loss(X, y)
    for _ in range(6):
        c.train(c.model.get_adapter(), epochs=2, lr=0.05)
    assert c.model.loss(X, y) < before


def test_base_weights_stay_frozen():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5))
    y = rng.normal(size=(200, 1))
    c = Client('N1', X, y, rank=4, d_hidden=12, d_out=1, task='reg')
    w1 = c.model.W1.copy()
    c.train(c.model.get_adapter(), epochs=3, lr=0.05)
    assert np.allclose(c.model.W1, w1)


def test_low_tier_client_sends_fewer_bytes():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(100, 5)), rng.normal(size=(100, 1))
    hi = Client('H', X, y, tier='hi', rank=16, d_out=1)
    lo = Client('L', X, y, tier='lo', rank=8, d_out=1)
    assert lo.uplink_bytes() < hi.uplink_bytes()


def test_server_round_accepts_mixed_ranks():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(150, 5)), rng.normal(size=(150, 1))
    cs = [Client('A', X, y, tier='hi', rank=16, d_hidden=16, d_out=1),
          Client('B', X, y, tier='lo', rank=8, d_hidden=16, d_out=1)]
    srv = Server(5, 16, 1, rank=16, task='reg')
    info = srv.round(cs, frac=1.0)
    assert set(info['selected']) == {'A', 'B'}
    assert srv.adapter['A'].shape[1] == 16


def test_drift_changes_input_distribution():
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(300, 4)), rng.normal(size=(300, 1))
    c = Client('N', X.copy(), y, rank=4, d_out=1)
    before = c.X.mean(axis=0).copy()
    c.apply_drift(1.0)
    assert c.drifted
    assert not np.allclose(c.X.mean(axis=0), before)
