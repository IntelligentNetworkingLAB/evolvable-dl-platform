# 세 가지 도메인 데이터를 만든다. seed 를 주면 같은 값이 나온다.
#
# 실측 트레이스가 아닌 합성 데이터이며, 도메인 구조(Zipf 인기도, 시간대 변동, 다중경로 채널)를
# 반영했다. 실측 CSV 로 바꿀 수 있도록 반환 형식을 맞춰 두었다.

import csv
import os

import numpy as np


def _diurnal(t, period):
    x = 2 * np.pi * (t % period) / period
    return 0.55 + 0.45 * np.sin(x - np.pi / 2) ** 2


# 콘텐츠 요청 트레이스
def content_requests(n_bs=6, n_content=120, n_slots=320, period=48, seed=1):
    rng = np.random.default_rng(seed)

    zipf = 1.0 / np.arange(1, n_content + 1) ** 1.15
    zipf = zipf / zipf.sum()

    # BS 마다 선호 순위를 섞어 non-IID 로 만든다
    order = np.stack([rng.permutation(n_content) for _ in range(n_bs)])
    local = np.zeros((n_bs, n_content))
    for b in range(n_bs):
        local[b, order[b]] = zipf
    local = 0.65 * local + 0.35 * zipf

    load = rng.uniform(28.0, 60.0, size=n_bs)
    counts = np.zeros((n_bs, n_slots, n_content), dtype=np.int32)
    for t in range(n_slots):
        d = _diurnal(t, period)
        for b in range(n_bs):
            lam = load[b] * d * local[b]
            counts[b, t] = rng.poisson(lam)
    return {'counts': counts, 'period': period, 'zipf': zipf}


def caching_features(trace, lags=4, warmup=8, norm=None):
    """(bs, t, content) 마다 다음 슬롯 요청량을 맞추는 문제로 바꾼다."""
    counts = trace['counts']
    n_bs, n_slots, n_content = counts.shape
    period = trace['period']
    zipf = trace['zipf']

    Xs, ys, ts, bs_id = [], [], [], []
    for t in range(warmup, n_slots - 1):
        d = 2 * np.pi * (t % period) / period
        for b in range(n_bs):
            hist = counts[b, t - lags + 1:t + 1]          # (lags, n_content)
            f = np.log1p(hist.T)                           # (n_content, lags)
            extra = np.stack([
                np.full(n_content, np.sin(d)),
                np.full(n_content, np.cos(d)),
                np.log1p(zipf * 1000.0),
                np.full(n_content, np.log1p(counts[b, t].sum())),
            ], axis=1)
            Xs.append(np.concatenate([f, extra], axis=1))
            ys.append(np.log1p(counts[b, t + 1]).reshape(-1, 1))
            ts.append(np.full(n_content, t))
            bs_id.append(np.full(n_content, b))

    X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    if norm is None:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    else:
        mu, sd = norm
    return {'X': (X - mu) / sd, 'y': y,
            't': np.concatenate(ts), 'bs': np.concatenate(bs_id),
            'counts': counts, 'mu': mu, 'sd': sd}


# 오프로딩 실행 로그
def offload_logs(n_nodes=6, n_per_node=900, seed=2):
    rng = np.random.default_rng(seed)
    Xs, ys, meta = [], [], []
    for b in range(n_nodes):
        n = n_per_node
        size = rng.lognormal(1.0, 0.7, n)                 # 입력 크기
        cx = rng.uniform(0, 1, n)                          # 난이도
        load = np.clip(rng.beta(2, 5, n) + 0.12 * b / n_nodes, 0, 1)
        dl = rng.uniform(20, 260, n)                       # 마감 ms
        prev = rng.uniform(0, 1, n)                        # 직전 성공률

        # 엣지 모델이 이 요청을 제대로 처리할 수 있는가 (숨은 규칙 + 잡음)
        logit = (2.4 - 3.1 * cx - 0.55 * np.log1p(size) - 1.5 * load
                 + 1.1 * prev + 0.004 * dl)
        p = 1.0 / (1.0 + np.exp(-logit))
        ok = (rng.uniform(size=n) < p).astype(np.int64)

        X = np.stack([np.log1p(size), cx, load, dl / 260.0, prev], axis=1)
        Xs.append(X)
        ys.append(ok)
        meta.append(np.stack([size, cx, load, dl], axis=1))

    X = np.concatenate(Xs)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    return {'X': (X - mu) / sd, 'y': np.concatenate(ys),
            'meta': np.concatenate(meta), 'n_nodes': n_nodes,
            'n_per_node': n_per_node}


# 채널 트레이스
def channel_traces(n_bs=6, n_sub=32, n_sym=700, n_tap=4, pilot_step=4,
                   snr_db=12.0, seed=3):
    rng = np.random.default_rng(seed)
    k = np.arange(n_sub)
    H = np.zeros((n_bs, n_sym, n_sub), dtype=np.complex128)
    for b in range(n_bs):
        delays = np.sort(rng.choice(np.arange(1, 7), size=n_tap, replace=False))  # 파일럿 간격 4 에서 지연 8 이상은 앨리어싱
        power = np.exp(-0.6 * np.arange(n_tap))
        power = power / power.sum()
        fd = rng.uniform(0.006, 0.02)                      # 정규화 도플러
        phase = rng.uniform(0, 2 * np.pi, n_tap)
        # 심볼마다 새로 뽑으면 시간 상관이 없어서 예측할 게 없다.
        # AR(1) 로 이어붙이고 도플러만큼 위상을 돌린다.
        rho = float(np.cos(2 * np.pi * fd))
        g = np.sqrt(power / 2) * (rng.normal(size=n_tap) +
                                  1j * rng.normal(size=n_tap))
        steer = np.exp(-2j * np.pi * np.outer(delays, k) / n_sub)
        for t in range(n_sym):
            innov = np.sqrt(power / 2) * (rng.normal(size=n_tap) +
                                          1j * rng.normal(size=n_tap))
            g = rho * g + np.sqrt(max(0.0, 1 - rho ** 2)) * innov
            gt = g * np.exp(1j * (2 * np.pi * fd * t + phase))
            H[b, t] = (gt[:, None] * steer).sum(axis=0)

    pilots = np.arange(0, n_sub, pilot_step)
    sigma = np.sqrt(10 ** (-snr_db / 10.0) / 2.0)
    noise = sigma * (rng.normal(size=(n_bs, n_sym, len(pilots))) +
                     1j * rng.normal(size=(n_bs, n_sym, len(pilots))))
    Y = H[:, :, pilots] + noise
    return {'H': H, 'Y': Y, 'pilots': pilots, 'n_sub': n_sub,
            'snr_db': snr_db, 'sigma': sigma}


def channel_features(tr):
    """파일럿 관측으로 다음 심볼의 전 서브캐리어 채널을 맞춘다."""
    H, Y = tr['H'], tr['Y']
    n_bs, n_sym, n_sub = H.shape
    X = np.concatenate([Y[:, :-1].real, Y[:, :-1].imag], axis=2)
    y = np.concatenate([H[:, 1:].real, H[:, 1:].imag], axis=2)
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1, y.shape[2])
    bs = np.repeat(np.arange(n_bs), n_sym - 1)
    sd = X.std() + 1e-8
    return {'X': X / sd, 'y': y, 'bs': bs, 'scale': sd,
            'n_sub': n_sub, 'n_bs': n_bs}


# CSV 로 떨어뜨리기 (저장소에 표본을 같이 올리기 위함)
def dump_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path
