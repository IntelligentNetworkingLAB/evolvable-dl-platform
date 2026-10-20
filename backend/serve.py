#!/usr/bin/env python3
# 데모용 HTTP 서버. 서버 자체는 표준 라이브러리만 쓴다.
#
# 프론트엔드(저장소 루트)를 정적으로 서빙하고 /api 아래에 학습 API 를 연다.
# 다른 프레임워크(FastAPI 등)로 바꿀 때는 edl/ 패키지는 그대로 두고 이 파일만 교체하면 된다.
#
#   python backend/serve.py            → http://127.0.0.1:8000
#   python backend/serve.py --port 9000 --host 0.0.0.0

import argparse
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edl.runner import Runner, task_catalog     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = '0.1.0'
RUNNER = Runner()
_build_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if '/api/' in (self.path or ''):
            sys.stderr.write('  %s %s\n' % (self.command, self.path))

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8'))
        except ValueError:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        if not self.path.startswith('/api/'):
            return super().do_GET()
        parts = [p for p in self.path.split('?')[0].strip('/').split('/') if p]
        try:
            if parts == ['api', 'health']:
                import numpy
                import torch
                return self._send({'ok': True, 'version': VERSION,
                                   'numpy': numpy.__version__,
                                   'torch': torch.__version__,
                                   'tasks': task_catalog()})
            if parts == ['api', 'tasks']:
                return self._send({'tasks': task_catalog()})
            if len(parts) == 3 and parts[:2] == ['api', 'run']:
                run = RUNNER.get(parts[2])
                if not run:
                    return self._send({'error': 'unknown run'}, 404)
                return self._send({**run.meta(), 'history': run.history,
                                   'events': run.events})
        except Exception as exc:                      # noqa: BLE001
            return self._send({'error': str(exc)}, 500)
        self._send({'error': 'not found'}, 404)

    def do_POST(self):
        parts = [p for p in self.path.split('?')[0].strip('/').split('/') if p]
        body = self._body()
        try:
            if parts == ['api', 'run']:
                task = body.get('task', 'caching')
                seed = int(body.get('seed', 0))
                rounds = int(body.get('rounds', 24))
                with _build_lock:                     # 데이터 생성이 무거워서 직렬화
                    run = RUNNER.create(task, seed=seed, rounds=rounds)
                return self._send({**run.meta(), 'history': run.history})

            if len(parts) == 4 and parts[:2] == ['api', 'run']:
                run = RUNNER.get(parts[2])
                if not run:
                    return self._send({'error': 'unknown run'}, 404)
                if parts[3] == 'step':
                    return self._send(run.step())
                if parts[3] == 'event':
                    return self._send(run.event(body.get('kind', 'drift')))
        except KeyError as exc:
            return self._send({'error': 'unknown task %s' % exc}, 400)
        except Exception as exc:                      # noqa: BLE001
            return self._send({'error': str(exc)}, 500)
        self._send({'error': 'not found'}, 404)

    def do_DELETE(self):
        parts = [p for p in self.path.strip('/').split('/') if p]
        if len(parts) == 3 and parts[:2] == ['api', 'run']:
            return self._send({'dropped': RUNNER.drop(parts[2])})
        self._send({'error': 'not found'}, 404)


def main():
    ap = argparse.ArgumentParser(description='진화형 딥러닝 모델 생성 플랫폼 데모 서버')
    # 컨테이너에서 띄울 때는 환경변수로 준다
    ap.add_argument('--host', default=os.environ.get('EDL_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int,
                    default=int(os.environ.get('EDL_PORT', '8000')))
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = 'http://%s:%d/' % ('127.0.0.1' if args.host == '0.0.0.0' else args.host,
                             args.port)
    print('데모      %s' % url)
    print('API       %sapi/health' % url)
    print('정적 경로 %s' % ROOT)
    print('종료      Ctrl+C')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n종료')
        srv.server_close()


if __name__ == '__main__':
    main()
