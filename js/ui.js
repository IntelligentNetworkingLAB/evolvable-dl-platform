// 화면 렌더링과 시연 진행.

const $  = s => document.querySelector(s);
const $$ = s => Array.prototype.slice.call(document.querySelectorAll(s));
const sleep = ms => new Promise(r => setTimeout(r, ms));

let running = false;   // 자동 재생 중
let token   = 0;   // 실행 취소 토큰
let pipeDone = false;   // 모델 생성 파이프라인 완료 여부

function log(kind, src, msg){
  const b = $('#logBody');
  const d = new Date();
  const ts = String(d.getHours()).padStart(2,'0') + ':' +
             String(d.getMinutes()).padStart(2,'0') + ':' +
             String(d.getSeconds()).padStart(2,'0');
  const row = document.createElement('div');
  row.className = 'lg-row ' + kind;
  row.innerHTML = `<span class="lg-t">${ts}</span><span class="lg-s">[${src}]</span><span class="lg-m"></span>`;
  row.querySelector('.lg-m').textContent = msg;
  b.appendChild(row);
  b.scrollTop = b.scrollHeight;
  while (b.children.length > 300) b.removeChild(b.firstChild);
}

function buildTabs(){
  const w = $('#taskTabs');
  w.innerHTML = '';
  TASK_ORDER.forEach(id => {
    const t = TASKS[id];
    const b = document.createElement('button');
    b.className = 'tasktab' + (id === Sim.task ? ' on' : '');
    b.dataset.id = id;
    b.innerHTML = `<span>${t.label}</span><span class="cn">${t.comp}</span>`;
    b.onclick = () => switchTask(id);
    w.appendChild(b);
  });
}

function buildStepper(){
  const w = $('#stepper');
  w.innerHTML = '';
  STEPS.forEach((s,i) => {
    const li = document.createElement('li');
    li.dataset.i = i;
    li.innerHTML = `<span class="sn">${i+1}</span><span class="st-n">${s.n}</span><span class="sx"></span>`;
    w.appendChild(li);
  });
}
function stepMark(i, cls, extra){
  const li = $(`#stepper li[data-i="${i}"]`); if (!li) return;
  li.className = cls || '';
  if (extra != null) li.querySelector('.sx').textContent = extra;
}
function stepReset(){ STEPS.forEach((s,i) => stepMark(i, '', '')); }

// 타자기 효과. 문자마다 setTimeout 을 걸면 타이머가 스로틀되는 환경에서
// 수십 초가 걸린다. 그래서 인터벌 하나로 두고 경과 시간으로 진행률을 낸다.
function typeNL(text, tk){
  const box = $('#nlSpec');
  const cur = document.createElement('span');
  cur.className = 'cursor';
  const DUR = 1500, t0 = performance.now();
  return new Promise(res => {
    const iv = setInterval(() => {
      if (tk !== token){ clearInterval(iv); res(); return; }
      const p = Math.min(1, (performance.now() - t0) / DUR);
      box.textContent = text.slice(0, Math.round(text.length * p));
      box.appendChild(cur);
      if (p >= 1){ clearInterval(iv); cur.remove(); res(); }
    }, 33);
  });
}
function renderSpec(){
  const t = Sim.T();
  let spec = t.spec;
  if (Sim.live && Sim.liveMeta){
    const m = Sim.liveMeta;
    const rows = m.nodes.reduce((a, n) => a + n.samples, 0);
    spec = [
      { k:'입력 데이터', v:`${rows.toLocaleString()} 행 · ${m.nodes.length}개 노드` },
      { k:'도메인',      v:t.spec[1].v },
      { k:'학습 설정',   v:`lr ${m.lr} · epoch ${m.epochs} · batch ${m.batch}` },
      { k:'비교 기준',   v:m.baseline },
    ];
  }
  $('#specGrid').innerHTML = spec.map(s =>
    `<div class="spec"><span>${s.k}</span><b>${s.v}</b></div>`).join('');
  $('#appTag').textContent = t.comp + ' ' + t.compName;
  $$('.chip').forEach(c => c.classList.toggle('active',
    c.dataset.c === ({'③':'3','④':'4','⑤':'5'}[t.comp] || '')));
}

function renderReco(anim){
  const t = Sim.T();
  if (Sim.live && Sim.liveMeta){
    const m = Sim.liveMeta;
    const rows = m.nodes.map(n => `
      <div class="cand">
        <span class="cand-n">${n.id}</span>
        <span class="cand-m">${n.tier} · ${(n.samples/1000).toFixed(1)}K행</span>
        <span class="cand-bar"><i style="width:${100 * n.rank / 16}%"></i></span>
        <span class="cand-p">r=${n.rank}</span>
      </div>`).join('');
    $('#recoBox').innerHTML = `
      <div class="reco-card">
        <div class="reco-h">실행 중인 모델 · ${m.model}</div>
        ${rows}
        <div class="hp">
          <div style="grid-column:1/3"><span>비교 기준</span><b>${m.baseline}</b></div>
          <div><span>Learning rate</span><b>${m.lr}</b></div>
          <div><span>Local epoch</span><b>${m.epochs}</b></div>
          <div><span>Batch</span><b>${m.batch}</b></div>
          <div><span>총 라운드</span><b>${m.rounds}</b></div>
        </div>
      </div>`;
    return;
  }
  const cands = t.candidates.map(c => `
    <div class="cand ${c.pick ? 'pick' : ''}">
      <span class="cand-n">${c.n}</span>
      <span class="cand-m">${c.m}</span>
      <span class="cand-bar"><i style="width:${anim ? c.s : 0}%"></i></span>
      <span class="cand-p">${c.s}%</span>
    </div>`).join('');
  $('#recoBox').innerHTML = `
    <div class="reco-card">
      <div class="reco-h">Model Architecture Dictionary → 후보 적합도</div>
      ${cands}
      <div class="hp">
        <div style="grid-column:1/3"><span>추천 손실함수</span><b>${t.loss}</b></div>
        ${t.hparams.map(h => `<div><span>${h.k}</span><b>${h.v}</b></div>`).join('')}
      </div>
    </div>`;
  if (!anim) setTimeout(() =>
    $$('#recoBox .cand-bar i').forEach((el,i) => el.style.width = t.candidates[i].s + '%'), 40);
}

const STLABEL = { idle:'대기', train:'로컬 학습', up:'어댑터 전송', drift:'DRIFT', new:'신규 합류', off:'미배포' };
function renderNodes(){
  const rows = Sim.nodes().map(n => {
    const sp = Sim.spec(n);
    const ls = Sim.localLoss[n.id];
    const lv = (ls && ls.length) ? ls[ls.length-1].toFixed(5) : '—';
    let st = 'off';
    if (Sim.deployed) st = (n.id === Sim.driftNode) ? 'drift' : 'idle';
    if (Sim.uiState && Sim.uiState[n.id]) st = Sim.uiState[n.id];
    const cls = st === 'drift' ? 'drift' : (n.joinsLater && Sim.joined && Sim.round < 3 ? 'newn' : '');
    return `<tr class="${cls}">
      <td class="mono"><span class="kind">${n.kind}</span>${n.nm}</td>
      <td>${n.hw}</td>
      <td class="mono">${Sim.deployed ? sp.model : '—'}</td>
      <td>${Sim.deployed ? `<span class="q ${sp.quant.toLowerCase()}">${sp.quant}</span>` : '—'}</td>
      <td class="mono">${Sim.deployed ? 'r=' + sp.rank : '—'}</td>
      <td>${Sim.deployed
          ? (sp.prune ? `<span class="prune"><i><b style="width:${sp.prune}%"></b></i>${sp.prune}%</span>`
                      : '<span class="prune none">—</span>')
          : '—'}</td>
      <td class="mono">${lv}</td>
      <td><span class="st ${st}">${STLABEL[st]}</span></td>
    </tr>`;
  }).join('');
  $('#nodeRows').innerHTML = rows;
}
function uiState(id, st){
  if (!Sim.uiState) Sim.uiState = {};
  Sim.uiState[id] = st;
}

function fmt(v, dec){ return dec === 0 ? Math.round(v).toString() : v.toFixed(dec); }

// 주 지표 하나만 크게 표시하고 나머지는 아래 목록에 둔다.
function renderKpis(flash){
  const t = Sim.T();
  const defs = Sim.kpiDefs(), mainId = Sim.mainKpiId();
  const main = defs.find(k => k.id === mainId) || defs[0];
  const rest = defs.filter(k => k.id !== main.id);

  const delta = k => {
    const v = Sim.cur(k.id);
    const improved = k.dir === 'up' ? v > k.from : v < k.from;
    const pct = k.from !== 0 ? Math.abs((v - k.from) / k.from * 100) : Math.abs(v - k.from);
    return {
      v, improved,
      cls: Sim.round === 0 ? 'flat' : (improved ? 'up' : 'down'),
      txt: Sim.round === 0 ? '시연 대기'
         : (k.dir === 'up' ? '▲ ' : '▼ ') + fmt(pct, 1) + '%'
    };
  };

  const d = delta(main);
  const prog = Math.max(0, Math.min(1,
    (d.v - main.from) / ((main.to - main.from) || 1)));
  $('#kpiHero').innerHTML = `
    <div class="hero ${Sim.round === 0 ? '' : (d.improved ? 'good' : 'bad')} ${flash ? 'flash' : ''}">
      <div class="hero-t">
        <span class="hero-l">${main.l}</span>
        <span class="hero-tag">${Sim.live ? '실측 · ' + Sim.liveMeta.model : '주 지표 · ' + t.comp + ' ' + t.compName}</span>
      </div>
      <div class="hero-v">${fmt(d.v, main.dec)}<span class="u">${main.u}</span></div>
      <div class="hero-d ${d.cls}">
        <span class="dl">${d.txt}</span>
        <span class="meta">시작 ${fmt(main.from, main.dec)}</span>
        ${Sim.baseline ? `<span class="meta">Baseline ${fmt(Sim.curBase(main.id), main.dec)}</span>` : ''}
        <span class="meta" style="margin-left:auto">목표 ${fmt(main.to, main.dec)}${main.u}</span>
      </div>
      <div class="hero-track"><i style="width:${(prog*100).toFixed(1)}%"></i><b style="left:100%"></b></div>
    </div>`;

  $('#kpiSub').innerHTML = rest.map(k => {
    const r = delta(k);
    return `<div class="sub">
      <span class="sub-l">${k.l}</span>
      <span class="sub-v">${fmt(r.v, k.dec)}<span class="u">${k.u}</span></span>
      <span class="sub-d ${r.cls}">${Sim.round === 0 ? '—' : r.txt}</span>
      <span class="sub-b">${Sim.baseline ? 'base ' + fmt(Sim.curBase(k.id), k.dec) : ''}</span>
    </div>`;
  }).join('');
}

// 정상일 때는 그리지 않는다.
function renderAlert(){
  const el = $('#alertStrip'); if (!el) return;
  const a = Sim.alert;
  if (!a){
    el.hidden = true; el.className = '';
    document.body.classList.remove('has-alert');
    return;
  }
  el.hidden = false;
  el.className = a.kind === 'info' ? 'info' : '';
  document.body.classList.add('has-alert');
  el.innerHTML = `
    <span class="al-k">${a.label}</span>
    <span class="al-n">${a.node}</span>
    <span class="al-t">${a.title}</span>
    <span class="al-s">${a.sub}</span>`;
}

function renderCharts(){
  const t = Sim.T();
  const k = Sim.kpiDefs().find(x => x.id === Sim.mainKpiId()) || Sim.kpiDefs()[0];
  $('#chartTitle').textContent = Sim.live ? (k.l + ' 추이 (라운드별)') : t.chartTitle;
  const ours = [k.from].concat(Sim.series[k.id]);
  const base = [k.from].concat(Sim.baseSeries[k.id]);
  drawKpiChart($('#kpiChart'), ours, base, {
    dec:k.dec, unit:k.u, showBase:Sim.baseline,
    target: (!Sim.live && t.id === 'channel') ? 30 : null
  });
  drawLossChart($('#lossChart'), Sim.globLoss, Sim.localLoss);
}

function renderComm(){
  const c = Sim.comm();
  $('#cmData').textContent = c.dataGB.toFixed(1) + ' GB';
  $('#cmFull').textContent = c.fullGB.toFixed(1) + ' GB';
  $('#cmLora').textContent = c.loraMB.toFixed(1) + ' MB';
  $('#cmSave').textContent = Sim.roundsUp ? c.saved.toFixed(2) + ' %' : '—';
  $('#commFill').style.width = Sim.roundsUp ? Math.min(100, c.saved) + '%' : '0%';
}

function renderVs(){
  const box = $('#vsMetric'), lab = $('.vs-t');
  if (Sim.live && Sim.liveMeta){
    lab.textContent = '실측 · ' + Sim.liveMeta.model;
    box.innerHTML = `기준 <b>${Sim.liveMeta.baseline}</b>`;
    return;
  }
  lab.textContent = 'vs 기준선';
  const v = Sim.T().vs;
  box.innerHTML =
    `${v.label} <b>${v.base.toFixed(3)}</b> → <b class="up">${Sim.vsNow().toFixed(3)}</b>`;
}

function renderMode(){
  const el = $('#modeBadge'); if (!el) return;
  if (Sim.replay){
    el.className = 'mode rec';
    el.textContent = '실측 기록';
    el.title = '서버에서 미리 돌린 학습 결과를 재생합니다. 지어낸 값이 아닙니다';
  } else if (Sim.live){
    el.className = 'mode live';
    el.textContent = '실측 · 실시간';
    el.title = '백엔드에서 지금 학습과 평가가 돌고 있습니다';
  } else {
    el.className = 'mode';
    el.textContent = '시뮬레이션';
    el.title = Api.ok ? '이 Task 는 백엔드 미지원이라 시뮬레이션으로 돕니다'
                      : '백엔드도 기록도 없어 시뮬레이션으로 돕니다';
  }
}

function lerpC(a, b, t){
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}

function renderApp(){
  const t = Sim.T(), w = $('#appView');
  const p = Sim.round / MAX_ROUND;

  if (t.view === 'cache'){
    const hit = Sim.cur('hit') / 100;
    const N = 80, nh = Math.round(N*hit);
    let cells = '';
    for (let i=0;i<N;i++){
      let c = 'miss';
      if (i < nh){ const m = i % 5; c = m === 4 ? 'lora' : (m === 3 ? 'kv' : 'hit'); }
      cells += `<div class="cbox ${c}"></div>`;
    }
    w.innerHTML = `<div class="av-h">SBS 엣지 캐시 점유 현황 · t+1 예측 배치 결과</div>
      <div class="av-b"><div class="cachegrid">${cells}</div>
      <div class="cachelg">
        <span><i style="background:#3f8a6c"></i>콘텐츠 히트</span>
        <span><i style="background:#3d8698"></i>KV 캐시</span>
        <span><i style="background:#7269a4"></i>모델·LoRA 가중치</span>
        <span><i style="background:#f4e8e6;box-shadow:inset 0 0 0 1px #e5cecb"></i>미스 → 백홀</span>
      </div></div>`;

  } else if (t.view === 'cascade'){
    const e = Sim.cur('edge'), be = Sim.curBase('edge');
    const bar = (ev) => `<span class="cas-bar">
        <i class="edge" style="width:${ev}%">${ev.toFixed(0)}% 엣지</i>
        <i class="cloud" style="width:${100-ev}%">${(100-ev).toFixed(0)}% 클라우드</i>
      </span>`;
    w.innerHTML = `<div class="av-h">추론 캐스케이드 · 엣지 종결 vs 클라우드 승격</div>
      <div class="av-b"><div class="cascade">
        <div class="cas-row"><span class="cl">Ours</span>${bar(e)}</div>
        <div class="cas-row"><span class="cl">${Sim.baselineName()}</span>${bar(be)}</div>
        <div class="cas-row" style="color:var(--tx3);font-size:10px;padding-top:2px">
          <span class="cl"></span>엣지에서 종결될수록 백홀·지연·에너지가 함께 내려갑니다</div>
      </div></div>`;

  } else if (t.view === 'heat'){
      const qq = Sim.round ? (1 - Math.exp(-3*p))/(1-Math.exp(-3)) : 0;
    let cells = '';
    for (let i=0;i<144;i++){
      const wave = 0.5 + 0.5*Math.sin(i*0.41 + (i/24)*0.9);
      const err  = Math.random()*(1-qq);
      const v = Math.max(0, Math.min(1, 0.15 + 0.5*wave*(1-0.62*err) + 0.35*qq));
      cells += `<div class="hcell" style="background:${lerpC(THEME.heatLo,THEME.heatHi,v)}"></div>`;
    }
    w.innerHTML = `<div class="av-h">서브캐리어 채널 이득 추정 맵 · t+1 (24 × 6 RB)</div>
      <div class="av-b"><div class="heat">${cells}</div>
      <div class="cachelg"><span>어두울수록 추정 오차 큼 · 라운드가 진행될수록 균질해집니다</span></div></div>`;

  } else {   // drone
    const m = Sim.cur('map') / 100;
    const conf = (0.42 + 0.55*m).toFixed(2);
    const boxes = [
      { x:14, y:22, w:22, h:34, c:'crit',  l:'person 0.' + Math.round(60+35*m) },
      { x:46, y:44, w:26, h:26, c:'warn',  l:'smoke 0.'  + Math.round(55+38*m) },
      { x:72, y:16, w:20, h:24, c:'',      l:'person 0.' + Math.round(52+40*m) }
    ].concat(m > 0.7 ? [{ x:33, y:64, w:17, h:22, c:'', l:'debris 0.'+Math.round(50+40*m) }] : []);
    const rep = Sim.round < 6
      ? '장면 판독 신뢰도 낮음 (0.'+Math.round(40+20*m)+'). 클라우드 VLM 승격 요청'
      : Sim.round < 15
      ? '건물 2층 창문에서 연기 확인. 인원 식별 시도 중. 일부 프레임 클라우드 승격.'
      : '3층 발코니에 성인 2명 추정 (신뢰도 '+conf+'). 좌측 외벽 균열 확대 중. 북측 계단 접근 권장. 온보드 종결, 클라우드 승격 없음.';
    w.innerHTML = `<div class="av-h">DRONE-01 온보드 VLM 판독 · 영상은 엣지를 떠나지 않음</div>
      <div class="av-b">
        <div class="dronefeed"><div class="grid"></div>
          ${boxes.map(b => `<div class="bbox ${b.c}" style="left:${b.x}%;top:${b.y}%;width:${b.w}%;height:${b.h}%"><span>${b.l}</span></div>`).join('')}
        </div>
        <div class="vlmout"><b>ON-DEVICE VLM 상황 보고</b>${rep}</div>
      </div>`;
  }
}

function renderAll(flash){
  $('#roundNum').textContent = Sim.round;
  const rp = $('#roundProgress');
  if (rp) rp.style.width = (Sim.round / MAX_ROUND * 100) + '%';
  renderAlert(); renderMode();
  renderKpis(flash); renderCharts(); renderComm(); renderVs();
  renderNodes(); renderApp();
}

async function runPipeline(tk){
  const t = Sim.T();
  stepReset();
  $('#recoBox').innerHTML = '<div class="reco-empty">모델 생성 파이프라인 실행 중…</div>';

  log('i','Platform', `Task 수신: ${t.comp} ${t.compName} / ${t.label}`);
  await typeNL(t.nl, tk); if (tk !== token) return false;

  const extras = [
    t.spec[0].v,
    '특징 18개 중 11개 선택',
    'Zoo 후보 ' + t.candidates.length + '종',
    'LoRA ' + t.hparams[0].v.replace('r=',''),
    '평균 압축 3.4×',
    t.candidates[0].n,
    Sim.nodes().length + ' 노드'
  ];

  for (let i = 0; i < STEPS.length - 1; i++){
    if (tk !== token) return false;
    stepMark(i, 'act', '…');
    log('i', STEPS[i].n.slice(0,14), STEPS[i].d);

    if (i === 5) renderReco(false);

    if (i === 6){   // 분산 엣지 배포
      if (Api.supports(Sim.task)){
        try {
          log('i','Backend', '백엔드 연결 · 데이터 적재와 클라이언트 분할');
          const meta = await Api.start(Sim.task, 0);
          Sim.beginLive(meta);
          log('ok','Backend',
            `${meta.model} · ${meta.nodes.length}개 클라이언트 · ` +
            meta.nodes.map(n => `${n.id}(${(n.samples/1000).toFixed(0)}K행)`).join(' '));
          log('i','Backend', `비교 기준: ${meta.baseline}`);
        } catch (e){
          log('warn','Backend', '백엔드 응답 없음. 기록을 찾아봅니다');
          Sim.live = false;
        }
      }
      if (!Sim.live){
        const rec = await Api.loadRecorded(Sim.task);
        if (rec){
          Sim.beginRecorded(rec);
          log('ok','Replay',
            `서버 학습 기록 재생 · ${rec.meta.model} · ${rec.rounds}라운드 · ` +
            `${rec.generated} 생성`);
          log('i','Replay', `비교 기준: ${rec.meta.baseline}`);
        }
      }
      Sim.deployed = true;
      Sim.uiState = {};
      Sim.nodes().forEach(n => Topo.setModel(n.id, Sim.spec(n).model.split('-').slice(0,2).join('-'), Sim.spec(n).quant));
      renderNodes();
      const ids = Sim.nodes().map(n => n.id);
      await new Promise(r => Topo.flySet(ids, 'down', r));
      log('ok','Deploy', '이종 엣지 차등 배포 완료: ' +
        Sim.nodes().map(n => `${n.nm}:${Sim.spec(n).quant}`).join(', '));
    } else {
      await sleep(STEPS[i].ms);
    }
    if (tk !== token) return false;
    stepMark(i, 'done', extras[i] || '완료');
  }

  stepMark(STEPS.length-1, 'act', 'R0');
  log('ok','Recommend', `글로벌 모델 ${t.candidates[0].n} 확정 · 손실함수 ${t.loss}`);
  log('i','Privacy', '데이터는 엣지에 잔류 · LoRA 어댑터만 업링크');
  pipeDone = true;
  renderAll();
  return true;
}

async function runRound(tk){
  let sel, result = null;

  if (Sim.live){
    // 실제 학습이 먼저 돌고, 참여한 노드를 받아서 그대로 그린다
    if (Sim.replay){
      result = Sim.nextRecorded();
      if (!result) Sim.live = false;
    } else {
      try {
        result = await Api.step();
      } catch (e){
        log('err','Backend', '라운드 실패. 시뮬레이션으로 전환합니다');
        Sim.live = false;
      }
    }
    if (tk !== token) return;
  }
  sel = result ? (result.selected || []) : Sim.select();

  sel.forEach(id => { Topo.setState(id, 'train'); uiState(id, 'train'); });
  renderNodes();
  await sleep(520); if (tk !== token) return;

  sel.forEach(id => { Topo.setState(id, 'up'); uiState(id, 'up'); });
  renderNodes();
  await new Promise(r => Topo.flySet(sel, 'up', r)); if (tk !== token) return;

  Topo.pulseCloud();
  if (result) Sim.ingest(result); else Sim.advance(sel);
  if (Sim.replay){
    const ev = Sim.recordedEventAt(Sim.round);
    if (ev) showEvent({ drift:'E1', join:'E2', cascade:'E3' }[ev.kind], ev.node, tk);
  }
  stepMark(STEPS.length-1, 'act', 'R' + Sim.round);
  renderAll(true);
  if (Sim.round % 4 === 0 || Sim.round === 1){
    const k = Sim.kpiDefs().find(x => x.id === Sim.mainKpiId()) || Sim.kpiDefs()[0];
    log('ok','Aggregator', `R${Sim.round} 집계 완료 · global loss ${Sim.globLoss[Sim.globLoss.length-1].toFixed(5)} · ${k.l} ${fmt(Sim.cur(k.id),k.dec)}${k.u}`);
  }

  await new Promise(r => Topo.flySet(sel, 'down', r)); if (tk !== token) return;
  sel.forEach(id => {
    const st = (id === Sim.driftNode) ? 'drift' : 'idle';
    Topo.setState(id, st); uiState(id, st);
  });
  renderNodes();
}

async function loop(tk){
  while (running && tk === token && Sim.round < MAX_ROUND){
    await runRound(tk);
    if (tk !== token) return;
    await sleep(200);
  }
  if (tk === token && Sim.round >= MAX_ROUND){
    running = false; setRunBtn();
    stepMark(STEPS.length-1, 'done', 'R' + MAX_ROUND);
    const t = Sim.T();
    log('ok','Platform', `${MAX_ROUND} 라운드 수렴 완료: ` +
      Sim.kpiDefs().map(k => `${k.l} ${fmt(Sim.cur(k.id),k.dec)}${k.u}`).join(' · '));
  }
}

function setRunBtn(){
  const b = $('#btnRun');
  b.textContent = running ? '⏸ 일시정지' : (pipeDone ? '▶ 계속' : '▶ 시연 시작');
  b.classList.toggle('running', running);
}

async function onRun(){
  if (running){ running = false; setRunBtn(); log('i','Control','일시정지'); return; }
  const tk = token;
  running = true; setRunBtn();
  if (!pipeDone){
    const ok = await runPipeline(tk);
    if (!ok || tk !== token) return;
  }
  loop(tk);
}

async function onStep(){
  if (running) return;
  const tk = token;
  if (!pipeDone){ await runPipeline(tk); return; }
  if (Sim.round < MAX_ROUND) await runRound(tk);
}

function resetAll(keepTask){
  token++; running = false; pipeDone = false;
  Api.reset();
  Sim.reset(keepTask ? Sim.task : Sim.task);
  Sim.uiState = {};
  Topo.flushAll(); Topo.init(); Topo.hideLate();
  stepReset(); setRunBtn();
  $('#nlSpec').textContent = '—';
  $('#recoBox').innerHTML = '<div class="reco-empty">Task를 선택하고 <b>시연 시작</b>을 누르면<br>모델 추천 결과가 표시됩니다.</div>';
  $$('.ev').forEach(b => { b.disabled = false; b.classList.remove('fired'); });
  renderSpec(); renderAll();
}

function switchTask(id){
  if (id === Sim.task) return;
  Sim.task = id;
  $$('.tasktab').forEach(b => b.classList.toggle('on', b.dataset.id === id));
  resetAll(true);
  log('i','Control', `Task 전환 → ${TASKS[id].comp} ${TASKS[id].compName}`);
}

// 이벤트가 났을 때 화면에서 벌어지는 일. 실시간과 기록 재생이 같이 쓴다.
async function showEvent(id, node, tk){
  if (!id || Sim.fired[id]) return;
  Sim.fired[id] = true;
  const btn = $('#ev' + id);
  if (btn){ btn.disabled = true; btn.classList.add('fired'); }

  if (id === 'E2'){
    Sim.alert = { kind:'info', node:node, label:'신규 노드',
      title:'저사양 노드 합류. 하드웨어 프로파일에 맞춰 모델 재추천',
      sub:'LoRA r=4 · INT4 · 구조적 pruning 62%', until: Sim.round + 3 };
    Topo.reveal('DRONE-03');
    const d3 = Sim.nodes().find(n => n.id === 'DRONE-03')
            || NODES.find(n => n.id === 'DRONE-03');
    Topo.setModel('DRONE-03',
      Sim.spec(d3).model.split('-').slice(0,2).join('-'), 'INT4');
  } else {
    Sim.driftNode = node;
    Sim.alert = id === 'E1'
      ? { kind:'alarm', node:node, label:'이상',
          title:'입력 분포 급변. Model Replacement Decision 실행 중',
          sub:'인근 셀 동시 영향 · 어댑터 재수렴 진행', until: Sim.round + 5 }
      : { kind:'alarm', node:node, label:'이상',
          title:'성능 열화 감지. 자율 재증류 실행 중',
          sub:'글로벌 어댑터로 로컬 모델 복원 · 2라운드 뒤 적용', until: Sim.round + 5 };
    Topo.setState(node, 'drift'); uiState(node, 'drift');
  }
  renderAll();

  log('evt','EVENT ' + id, EVENTS[id].name);
  for (const [k, src, m] of EVENTS[id].log){
    await sleep(620);
    if (tk !== token) return;
    log(k, src, m);
  }
  if (id === 'E2'){
    await new Promise(r => Topo.flySet(['DRONE-03'], 'down', r));
    if (tk !== token) return;
    Topo.setState('DRONE-03','idle'); uiState('DRONE-03','idle');
  }
  if (tk === token) renderNodes();
}

async function fireEvent(id){
  if (!Sim.deployed){ log('warn','Control','배포 이후에 이벤트를 주입할 수 있습니다'); return; }
  if (Sim.replay){
    const ev = (Sim.recorded.events || [])
      .find(e => ({ drift:'E1', join:'E2', cascade:'E3' })[e.kind] === id);
    log('i','Replay', ev
      ? `${id} 은 기록된 R${ev.round} 에 자동으로 발생합니다`
      : `${id} 은 이 기록에 없습니다`);
    return;
  }
  const tk = token;   // 진행 중에 초기화나 Task 전환이 오면 바로 빠진다
  const E = EVENTS[id];

  if (Sim.live){
    if (Sim.fired[id]) return;
    Sim.fired[id] = true;
    const kind = { E1:'drift', E2:'join', E3:'cascade' }[id];
    let res;
    try {
      res = await Api.event(kind);
    } catch (e){
      log('err','Backend', '이벤트 주입 실패'); Sim.fired[id] = false; return;
    }
    if (tk !== token) return;
    if (id === 'E2'){
      Sim.alert = { kind:'info', node:res.node, label:'신규 노드',
        title:'저사양 노드 합류. 하드웨어 프로파일에 맞춰 모델 재추천',
        sub:'LoRA r=4 · INT4 · 구조적 pruning 62%', until: Sim.round + 3 };
    } else {
      Sim.driftNode = res.node;
      Sim.alert = id === 'E1'
        ? { kind:'alarm', node:res.node, label:'이상',
            title:'입력 분포 급변. Model Replacement Decision 실행 중',
            sub:'인근 셀 동시 영향 · 어댑터 재수렴 진행', until: Sim.round + 5 }
        : { kind:'alarm', node:res.node, label:'이상',
            title:'성능 열화 감지. 자율 재증류 실행 중',
            sub:'글로벌 어댑터로 로컬 모델 복원 · 2라운드 뒤 적용', until: Sim.round + 5 };
    }
  } else if (!Sim.fire(id)) {
    return;
  }
  const btn = $('#ev' + id);
  btn.disabled = true; btn.classList.add('fired');

  log('evt','EVENT ' + id, E.name);

  if (id === 'E2'){
    Topo.reveal('DRONE-03');
    const d3 = Sim.nodes().find(n => n.id === 'DRONE-03')
            || NODES.find(n => n.id === 'DRONE-03');
    Topo.setModel('DRONE-03',
      Sim.spec(d3).model.split('-').slice(0,2).join('-'), 'INT4');
    if (!Sim.live){
      Sim.localLoss['DRONE-03'] =
        new Array(Sim.globLoss.length).fill(Sim.globLoss[Sim.globLoss.length-1] * 1.6);
    }
  } else {
    const node = (Sim.live && Sim.alert) ? Sim.alert.node : E.node;
    Topo.setState(node, 'drift'); uiState(node, 'drift');
  }
  renderNodes(); renderAll();

  for (const [k, s, m] of E.log){
    await sleep(620);
    if (tk !== token) return;
    log(k, s, m);
  }
  if (id === 'E2'){
    await new Promise(r => Topo.flySet(['DRONE-03'], 'down', r));
    if (tk !== token) return;
    Topo.setState('DRONE-03','idle'); uiState('DRONE-03','idle');
  }
  if (tk !== token) return;
  renderNodes();
  if (!running && Sim.round < MAX_ROUND){
    log('i','Control','▶ 계속을 눌러 재수렴 과정을 확인하세요');
  }
}

function applyTheme(name){
  const n = setTheme(name);
  // 노드 위 모델 라벨은 인라인 fill 이라 다시 칠해야 한다
  if (Sim.deployed){
    Sim.nodes().forEach(x => {
      const sp = Sim.spec(x);
      Topo.setModel(x.id, sp.model.split('-').slice(0,2).join('-'), sp.quant);
    });
  }
  renderAll();
  return n;
}

function cycleTheme(){
  const i = THEME_LIST.findIndex(t => t[0] === THEME_NAME);
  const next = THEME_LIST[(i + 1) % THEME_LIST.length];
  applyTheme(next[0]);
  log('i','Control', '테마 → ' + next[1]);
}

window.addEventListener('DOMContentLoaded', async () => {
  Sim.reset('caching');
  buildTabs(); buildStepper();
  applyTheme(savedTheme());
  Topo.init(); Topo.hideLate();
  renderSpec(); renderAll(); setRunBtn();

  $('#btnRun').onclick   = onRun;
  $('#btnStep').onclick  = onStep;
  $('#btnReset').onclick = () => { resetAll(true); log('i','Control','초기화 완료'); };
  $('#logClear').onclick = () => $('#logBody').innerHTML = '';
  $('#baseToggle').onchange = e => {
    Sim.baseline = e.target.checked;
    renderAll();
    log('i','Control', 'Baseline 비교 ' + (Sim.baseline ? 'ON, 기준 ' + Sim.baselineName() : 'OFF'));
  };
  ['E1','E2','E3'].forEach(id => $('#ev'+id).onclick = () => fireEvent(id));

  window.addEventListener('resize', () => renderCharts());

  // 키보드 단축키
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (e.code === 'Space'){ e.preventDefault(); onRun(); return; }
    if (e.key === 'ArrowRight'){ e.preventDefault(); onStep(); return; }
    if (k === 'r'){ resetAll(true); log('i','Control','초기화 완료'); return; }
    if (k === 'b'){ const t = $('#baseToggle'); t.checked = !t.checked;
                    t.dispatchEvent(new Event('change')); return; }
    if (k >= '1' && k <= '4'){ switchTask(TASK_ORDER[+k - 1]); return; }
    if (k === 't'){ cycleTheme(); return; }
    const evk = { q:'E1', w:'E2', e:'E3' }[k];
    if (evk) fireEvent(evk);
  });

  await Api.probe();
  renderMode();
  if (Api.ok){
    log('ok','Backend', `백엔드 연결됨 · numpy ${Api.info.numpy} · 실측 지원 ` +
      Api.info.tasks.map(t => t.id).join(', '));
  } else {
    log('warn','Backend', '실시간 백엔드 없음. 기록이 있으면 재생합니다 ' +
      '(python backend/serve.py 로 실시간 모드)');
  }
  log('ok','Platform','진화형 딥러닝 모델 생성 플랫폼 · 통합 빌드 기동');
  log('i','Fleet', `분산 엣지 플릿 ${Sim.nodes().length}개 노드 온라인 (MBS·SBS·UAA·Drone·UE·CubeSat)`);
  log('i','Control','Task를 고르고 [시연 시작]을 누르세요');
});
