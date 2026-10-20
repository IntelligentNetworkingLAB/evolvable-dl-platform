#!/usr/bin/env python3
# 세 도메인 데이터를 CSV 로 저장한다.
#
# 데모는 실행할 때마다 메모리에서 데이터를 만들어 쓰므로 이 파일 없이도 동작한다.
# 저장소에 표본을 포함해 데이터 형태를 확인하고, 실측 트레이스로 바꿀 때 맞춰야 할
# 컬럼을 보이는 용도다.
#
#   python backend/make_data.py                 → data/ 에 표본 저장
#   python backend/make_data.py --full          → 전체 크기로 저장

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edl.datasets import (channel_traces, content_requests,  # noqa: E402
                          dump_csv, offload_logs)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data')


def write_requests(limit):
    tr = content_requests(n_bs=6, n_content=80, n_slots=220, seed=1)
    c = tr['counts']
    rows = []
    for b in range(c.shape[0]):
        for t in range(c.shape[1]):
            for k in np.nonzero(c[b, t])[0]:
                rows.append((f'BS-{b + 1:02d}', t, int(k), int(c[b, t, k])))
                if limit and len(rows) >= limit:
                    return dump_csv(os.path.join(OUT, 'content_requests.csv'),
                                    ['bs_id', 'slot', 'content_id', 'requests'],
                                    rows)
    return dump_csv(os.path.join(OUT, 'content_requests.csv'),
                    ['bs_id', 'slot', 'content_id', 'requests'], rows)


def write_offload(limit):
    ds = offload_logs(n_nodes=6, n_per_node=1600, seed=2)
    m, y = ds['meta'], ds['y']
    n = min(len(y), limit) if limit else len(y)
    rows = [(f'NODE-{i // 1600 + 1:02d}', round(float(m[i, 0]), 4),
             round(float(m[i, 1]), 4), round(float(m[i, 2]), 4),
             round(float(m[i, 3]), 1), int(y[i])) for i in range(n)]
    return dump_csv(os.path.join(OUT, 'offload_logs.csv'),
                    ['node_id', 'input_size', 'complexity', 'node_load',
                     'deadline_ms', 'edge_ok'], rows)


def write_channel(limit):
    tr = channel_traces(n_bs=6, n_sub=32, n_sym=520, seed=3)
    H, Y, p = tr['H'], tr['Y'], tr['pilots']
    rows = []
    for b in range(H.shape[0]):
        for t in range(H.shape[1]):
            for j, k in enumerate(p):
                rows.append((f'BS-{b + 1:02d}', t, int(k),
                             round(float(Y[b, t, j].real), 5),
                             round(float(Y[b, t, j].imag), 5),
                             round(float(H[b, t, k].real), 5),
                             round(float(H[b, t, k].imag), 5)))
                if limit and len(rows) >= limit:
                    break
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    return dump_csv(os.path.join(OUT, 'channel_pilots.csv'),
                    ['bs_id', 'symbol', 'subcarrier', 'pilot_re', 'pilot_im',
                     'true_re', 'true_im'], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='전체 크기로 저장')
    args = ap.parse_args()
    limit = None if args.full else 20000

    for fn in (write_requests, write_offload, write_channel):
        path = fn(limit)
        size = os.path.getsize(path)
        print('%-46s %8.1f KB' % (os.path.relpath(path, os.path.dirname(OUT)),
                                  size / 1024))


if __name__ == '__main__':
    main()
