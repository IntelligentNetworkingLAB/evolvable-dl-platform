#!/usr/bin/env python3
# 학습을 끝까지 돌려서 결과를 파일로 남긴다.
#
# 모델이 커지면 브라우저에서 라운드를 실시간으로 돌리기 어렵다. 미리 돌려 기록을 남겨 두면
# 화면은 그 기록을 재생한다. 재생되는 값은 모두 실제 학습 결과다.
#
#   python backend/run_experiment.py                    전체 태스크
#   python backend/run_experiment.py --task caching     하나만
#   python backend/run_experiment.py --rounds 40 --seed 3
#   python backend/run_experiment.py --no-events        이벤트 없이
#
# 결과는 data/runs/<task>.js 로 저장된다. index.html 을 file:// 로 열면 fetch 가
# 막히므로 JSON 대신 script 태그로 읽을 수 있는 js 로 쓴다.

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edl.runner import Run  # noqa: E402
from edl.tasks import REGISTRY  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'runs')

# 기본 이벤트 시점
DEFAULT_EVENTS = [(10, 'drift'), (14, 'join'), (18, 'cascade')]


def record(task_id, rounds, seed, events):
    t0 = time.time()
    run = Run(task_id, seed=seed, rounds=rounds)
    plan = dict(events)
    fired = []

    while run.task.round < rounds:
        nxt = run.task.round + 1
        if nxt in plan:
            info = run.event(plan[nxt])
            fired.append({'round': nxt, 'kind': plan[nxt], 'node': info['node']})
            print('    R%-2d  이벤트 %s -> %s' % (nxt, plan[nxt], info['node']))
        run.step()

    k = run.history[-1]['kpis'][0]
    print('    R%-2d  %s %.2f%s (기준 %.2f)  %.1fs'
          % (rounds, k['label'], k['value'], k['unit'], k['baseline'],
             time.time() - t0))

    return {
        'task': task_id,
        'seed': seed,
        'rounds': rounds,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'meta': run.meta(),
        'history': run.history,
        'events': fired,
    }


def write(payload):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, payload['task'] + '.js')
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('// %s · seed %d · %d라운드 · 자동 생성물, 직접 고치지 말 것\n'
                % (payload['generated'], payload['seed'], payload['rounds']))
        f.write('window.EDL_RUNS=window.EDL_RUNS||{};\n')
        f.write('window.EDL_RUNS[%s]=%s;\n'
                % (json.dumps(payload['task']), body))
    return path


def main():
    ap = argparse.ArgumentParser(description='학습을 돌려 재생용 기록을 남긴다')
    ap.add_argument('--task', choices=sorted(REGISTRY), help='비우면 전체')
    ap.add_argument('--rounds', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-events', action='store_true')
    args = ap.parse_args()

    events = [] if args.no_events else DEFAULT_EVENTS
    tasks = [args.task] if args.task else sorted(REGISTRY)

    for tid in tasks:
        print('%s' % tid)
        path = write(record(tid, args.rounds, args.seed, events))
        print('    -> %s  %.0f KB\n'
              % (os.path.relpath(path), os.path.getsize(path) / 1024))


if __name__ == '__main__':
    main()
