// FL 라운드 시뮬레이션. 상태와 수식만 두고 DOM 은 건드리지 않는다.
// 진행 순서를 다루는 쪽은 ui.js.

const Sim = {

  task:'caching', round:0, deployed:false, baseline:false,
  dip:0,   // 진화 이벤트 충격량 (0=정상)
  fired:{ E1:false, E2:false, E3:false },
  joined:false,   // DRONE-03 합류 여부
  driftNode:null,
  series:{}, baseSeries:{}, globLoss:[], localLoss:{},
  alert:null,   // 진행 중인 이상 상황 (스트립에 노출)
  loraMB:0, roundsUp:0, lastSel:[],

  // 백엔드가 붙었을 때만 채워진다. 없으면 전부 비어 있고 아래 시뮬레이션이 돈다.
  live:false, liveKpis:null, liveNodes:null, liveComm:null, liveMeta:null,
  replay:false, recorded:null, replayAt:0,

  reset(taskId){
    if (taskId) this.task = taskId;
    this.round = 0; this.deployed = false; this.dip = 0;
    this.fired = { E1:false, E2:false, E3:false };
    this.joined = false; this.driftNode = null; this.alert = null;
    this.live = false; this.liveKpis = null; this.liveNodes = null;
    this.liveComm = null; this.liveMeta = null;
    this.replay = false; this.recorded = null; this.replayAt = 0;
    this.series = {}; this.baseSeries = {};
    this.globLoss = []; this.localLoss = {};
    this.loraMB = 0; this.roundsUp = 0; this.lastSel = [];
    this.T().kpis.forEach(k => { this.series[k.id] = []; this.baseSeries[k.id] = []; });
    this.nodes().forEach(n => this.localLoss[n.id] = []);
  },

  T(){ return TASKS[this.task]; },

  kpiDefs(){ return this.liveKpis || this.T().kpis; },
  mainKpiId(){ return this.liveKpis ? this.liveKpis[0].id : this.T().mainKpi; },
  baselineName(){
    return (this.live && this.liveMeta) ? this.liveMeta.baseline : this.T().baseline;
  },

  // 백엔드 실행을 붙인다
  beginLive(meta){
    this.reset(this.task);
    this.live = true;
    this.liveMeta = meta;
    const h0 = meta.history[0];
    this.liveKpis = h0.kpis.map(k => ({
      id:k.id, l:k.label, u:k.unit, dec:k.dec, dir:k.dir,
      from:k.value, base:k.baseline, to:k.baseline
    }));
    this.series = {}; this.baseSeries = {};
    this.liveKpis.forEach(k => { this.series[k.id] = []; this.baseSeries[k.id] = []; });
    this.liveNodes = meta.nodes;
    this.liveComm = h0.comm;
    this.localLoss = {};
    meta.nodes.forEach(n => this.localLoss[n.id] = []);
    this.deployed = true;
  },

  // 서버에서 기록해 둔 학습 결과를 재생한다. 값은 전부 실제 학습 결과다.
  beginRecorded(rec){
    this.beginLive(Object.assign({}, rec.meta, { history: [rec.history[0]] }));
    this.replay = true;
    this.recorded = rec;
    this.replayAt = 1;
  },

  nextRecorded(){
    if (!this.recorded) return null;
    if (this.replayAt >= this.recorded.history.length) return null;
    return this.recorded.history[this.replayAt++];
  },

  recordedEventAt(round){
    if (!this.recorded) return null;
    return (this.recorded.events || []).find(e => e.round === round) || null;
  },

  // 백엔드 라운드 결과를 차트가 쓰는 구조에 넣는다
  ingest(r){
    this.round = r.round;
    r.kpis.forEach(k => {
      if (!this.series[k.id]) { this.series[k.id] = []; this.baseSeries[k.id] = []; }
      this.series[k.id].push(k.value);
      this.baseSeries[k.id].push(k.baseline == null ? k.value : k.baseline);
    });
    this.globLoss.push(r.loss);
    if (r.nodes) this.liveNodes = r.nodes;
    if (r.comm) this.liveComm = r.comm;
    (this.liveNodes || []).forEach(n => {
      if (!this.localLoss[n.id]) this.localLoss[n.id] = [];
      const a = this.localLoss[n.id];
      const v = (r.client_loss && r.client_loss[n.id] != null)
        ? r.client_loss[n.id] : (a.length ? a[a.length - 1] : r.loss);
      a.push(v);
    });
    this.lastSel = r.selected || [];
    if (this.alert && this.alert.until != null && this.round >= this.alert.until){
      this.alert = null;
      this.driftNode = null;
    }
  },

  // 지금 플릿에 붙어 있는 노드
  nodes(){
    if (this.live && this.liveNodes){
      return this.liveNodes.map(n => {
        const d = NODES.find(x => x.id === n.id) || {};
        return Object.assign({}, d, n, { nm:n.id, kind:d.kind || '', hw:d.hw || '' });
      });
    }
    return NODES.filter(n => !n.joinsLater || this.joined);
  },

  spec(n){
    const tier = n.tier;
    if (this.live && n.rank != null){
      // 백엔드가 실제로 적용한 값
      return { model: (this.liveMeta && this.liveMeta.model) || '—',
               quant: QUANT[tier], rank: n.rank,
               prune: tier === 'xlo' ? 62 : tier === 'lo' ? 38
                    : tier === 'mid' ? 18 : 0 };
    }
    const t = this.T();
    return {
      model: t.models[tier],
      quant: QUANT[tier],
      rank : RANK[tier],
      prune: tier === 'xlo' ? 62 : tier === 'lo' ? 38 : tier === 'mid' ? 18 : 0
    };
  },

  // 수렴 곡선
  ease(p){
    p = Math.max(0, Math.min(1, p));
    const K = 3.0;
    return (1 - Math.exp(-K*p)) / (1 - Math.exp(-K));
  },

  // dip 이 클수록 목표에서 뒤로 밀린다
  kpiOurs(k, round, dip){
    const eff = this.ease(round/MAX_ROUND) * (1 - (dip||0));
    const v = k.from + (k.to - k.from) * eff;
    const jit = (k.to - k.from) * 0.012 * (Math.random()-0.5);
    return round === 0 ? k.from : v + jit;
  },

  // baseline 은 개선폭이 작고 흔들리기만 한다
  kpiBase(k, round){
    const eff = this.ease(round/MAX_ROUND);
    const v = k.from + (k.base - k.from) * eff;
    const jit = (k.base - k.from) * 0.05 * (Math.random()-0.5);
    return round === 0 ? k.from : v + jit;
  },

  // 학습 곡선 모양
  gloss(round, dip){
    const L0 = 0.0262, FL = 0.00082;
    const base = FL + (L0-FL) * Math.exp(-2.7 * round/MAX_ROUND);
    return base * (1 + (dip||0)*4.2) * (1 + 0.05*(Math.random()-0.5));
  },

  // 이번 라운드에 참여할 노드 뽑기
  select(){
    const av = this.nodes().filter(n => n.id !== 'MBS-01');
    const k = Math.max(3, Math.round(av.length * 0.72));
    const sh = av.slice().sort(() => Math.random()-0.5).slice(0, k).map(n => n.id);
    // drift 난 노드는 재수렴을 보여줘야 하니 반드시 넣는다
    if (this.driftNode && sh.indexOf(this.driftNode) < 0) sh[0] = this.driftNode;
    this.lastSel = sh;
    return sh;
  },

  advance(sel){
    if (this.round >= MAX_ROUND) return false;
    this.round++;
    this.dip *= 0.42;   // 이벤트 충격 자연 회복
    if (this.dip < 0.005) { this.dip = 0; this.driftNode = null; }
    // 회복되면 스트립을 내린다
    if (this.alert){
      if (this.alert.kind === 'info'){
        if (this.round >= this.alert.until) this.alert = null;
      } else if (this.dip < 0.04){
        this.alert = null;
      }
    }

    this.T().kpis.forEach(k => {
      this.series[k.id].push(this.kpiOurs(k, this.round, this.dip));
      this.baseSeries[k.id].push(this.kpiBase(k, this.round));
    });

    const g = this.gloss(this.round, this.dip);
    this.globLoss.push(g);
    this.nodes().forEach(n => {
      if (!this.localLoss[n.id]) this.localLoss[n.id] = [];
      const skew = n.tier === 'hi' ? 0.86 : n.tier === 'mid' ? 1.0 : n.tier === 'lo' ? 1.16 : 1.34;
      let v = g * skew * (1 + 0.09*(Math.random()-0.5));
      if (n.id === this.driftNode) v *= 2.4;   // drift 노드 손실 급등
      if (sel.indexOf(n.id) < 0) v = this.localLoss[n.id].length
        ? this.localLoss[n.id][this.localLoss[n.id].length-1] : v;
      this.localLoss[n.id].push(v);
    });

    this.loraMB += this.T().loraMB * sel.length;
    this.roundsUp++;
    return true;
  },

  fire(id){
    if (this.fired[id] || !this.deployed) return false;
    this.fired[id] = true;
    const E = EVENTS[id];
    this.dip = E.penalty;
    if (id === 'E2'){
      this.joined = true; this.localLoss['DRONE-03'] = [];
      this.alert = { kind:'info', node:'DRONE-03', label:'신규 노드',
        title:'저사양 노드 합류. 하드웨어 프로파일에 맞춰 모델 재추천',
        sub:'구조적 pruning 62% + INT4 · LoRA r=4', until:this.round + 3 };
    } else {
      this.driftNode = E.node;
      this.alert = id === 'E1'
        ? { kind:'alarm', node:E.node, label:'이상',
            title:'입력 분포 급변. Model Replacement Decision 실행 중',
            sub:'PSI 0.34 (임계 0.20) · 전체 재학습 4.2h → 어댑터 교체 3.5min' }
        : { kind:'alarm', node:E.node, label:'이상',
            title:'클라우드 승격률 폭주. 자율 재증류 실행 중',
            sub:'승격률 13% → 41% · Cloud teacher → Edge student' };
    }
    return true;
  },

  cur(kpiId){
    const s = this.series[kpiId];
    const k = this.kpiDefs().find(x => x.id === kpiId);
    return (s && s.length) ? s[s.length-1] : (k ? k.from : 0);
  },
  curBase(kpiId){
    const s = this.baseSeries[kpiId];
    const k = this.kpiDefs().find(x => x.id === kpiId);
    if (s && s.length) return s[s.length-1];
    return k ? (k.base != null ? k.base : k.from) : 0;
  },

  // 목표 성능치 기준
  vsNow(){
    const v = this.T().vs;
    const eff = this.ease(this.round/MAX_ROUND) * (1 - this.dip);
    return v.base + (v.target - v.base) * eff;
  },

  comm(){
    if (this.live && this.liveComm){
      const c = this.liveComm;
      return { dataGB: c.data_bytes / 1e9,
               fullGB: c.full_bytes_per_round / 1e9,
               loraMB: c.adapter_bytes_this_round / 1e6,
               saved: c.saving_vs_data_pct };
    }
    const t = this.T(), n = this.nodes().length;
    const dataGB = t.dataGB * n;
    const fullGB = t.fullGB * n * Math.max(1, this.roundsUp);
    const loraMB = this.loraMB;
    const saved  = fullGB > 0 ? (1 - (loraMB/1024) / fullGB) * 100 : 0;
    return { dataGB, fullGB, loraMB, saved:Math.max(0, saved) };
  }
};
