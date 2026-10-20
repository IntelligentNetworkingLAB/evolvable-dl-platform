// 백엔드 연결. 백엔드가 없으면 시뮬레이션으로 동작한다.
// 연결에 실패해도 예외를 위로 던지지 않는다.

const Api = {
  base: (location.protocol === 'file:') ? 'http://127.0.0.1:8000' : '',
  ok: false,
  info: null,
  runId: null,
  meta: null,

  async _fetch(path, opt, ms){
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), ms || 20000);
    try {
      const res = await fetch(this.base + path,
        Object.assign({ signal: ac.signal }, opt || {}));
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  },

  _post(path, body){
    return this._fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  },

  async probe(){
    try {
      this.info = await this._fetch('/api/health', null, 1500);
      this.ok = !!(this.info && this.info.ok);
    } catch (e) {
      this.ok = false; this.info = null;
    }
    return this.ok;
  },

  supports(taskId){
    return this.ok && !!this.info.tasks.find(t => t.id === taskId);
  },

  async start(taskId, seed){
    const r = await this._post('/api/run', { task: taskId, seed: seed || 0 });
    this.runId = r.run_id;
    this.meta = r;
    return r;
  },

  step(){
    return this._post('/api/run/' + this.runId + '/step');
  },

  event(kind){
    return this._post('/api/run/' + this.runId + '/event', { kind: kind });
  },

  reset(){
    this.runId = null;
    this.meta = null;
  },

  // 서버에서 미리 돌려둔 학습 기록. fetch 는 file:// 에서 막히므로
  // script 태그로 읽는다. 없으면 null 을 주고 시뮬레이션으로 넘어간다.
  loadRecorded(taskId){
    return new Promise(resolve => {
      const have = window.EDL_RUNS && window.EDL_RUNS[taskId];
      if (have) return resolve(have);
      const el = document.createElement('script');
      el.src = 'data/runs/' + taskId + '.js';
      el.onload = () => resolve((window.EDL_RUNS || {})[taskId] || null);
      el.onerror = () => resolve(null);
      document.head.appendChild(el);
    });
  }
};
