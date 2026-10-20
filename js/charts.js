// canvas 차트. 외부 라이브러리는 쓰지 않는다.

const C = {
  grid:THEME.grid, tx:THEME.ink3,
  ours:THEME.ours, base:THEME.base, target:THEME.target,
  fill:THEME.fill, node:THEME.series
};

// 뷰포트가 낮으면 차트를 줄여서 패널이 넘치지 않게 한다
function chartH(kind){
  const h = window.innerHeight;
  // 이상 스트립이 먹은 만큼 돌려준다
  const a = document.body.classList.contains('has-alert') ? 22 : 0;
  if (kind === 'kpi')  return (h >= 1000 ? 156 : h >= 940 ? 146 : h >= 840 ? 116 : 96) - a;
  return (h >= 1000 ? 94 : h >= 940 ? 90 : 66) - a;
}

// dpr 반영해서 캔버스 크기를 잡는다
function prep(cv, cssH){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth || 600;
  const h = cssH;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)){
    cv.width  = Math.round(w*dpr);
    cv.height = Math.round(h*dpr);
  }
  cv.style.height = h + 'px';
  const g = cv.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  return { g, w, h };
}

function niceBounds(vals){
  let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  if (lo === hi){ lo -= 1; hi += 1; }
  const pad = (hi-lo)*0.18;
  return { lo: lo-pad, hi: hi+pad };
}

function fmtNum(v, dec){
  return dec === 0 ? Math.round(v).toString() : v.toFixed(dec);
}

// KPI 추이. ours 와 baseline 을 겹쳐 그린다
function drawKpiChart(cv, ours, base, opt){
  const H = chartH('kpi');
  const { g, w, h } = prep(cv, H);
  const PL = 46, PR = 12, PT = 12, PB = 22;
  const iw = w - PL - PR, ih = h - PT - PB;

  const all = ours.concat(opt.showBase ? base : []);
  if (!all.length){
    g.fillStyle = C.tx; g.font = '11px sans-serif'; g.textAlign = 'center';
    g.fillText('시연을 시작하면 KPI 추이가 표시됩니다', w/2, h/2);
    return;
  }
  const b = niceBounds(all.concat(opt.target != null ? [opt.target] : []));
  const X = i => PL + (ours.length<2 ? 0 : iw * i/(MAX_ROUND));
  const Y = v => PT + ih - ih*(v-b.lo)/(b.hi-b.lo);

  g.font = '9px Consolas, monospace'; g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let i=0;i<=4;i++){
    const v = b.lo + (b.hi-b.lo)*i/4, y = Y(v);
    g.strokeStyle = C.grid; g.lineWidth = 1;
    g.beginPath(); g.moveTo(PL,y); g.lineTo(w-PR,y); g.stroke();
    g.fillStyle = C.tx; g.fillText(fmtNum(v,opt.dec), PL-6, y);
  }
  g.textAlign='center'; g.textBaseline='top';
  for (let r=0;r<=MAX_ROUND;r+=4){
    const x = PL + iw*r/MAX_ROUND;
    g.strokeStyle = C.grid; g.beginPath(); g.moveTo(x,PT); g.lineTo(x,PT+ih); g.stroke();
    g.fillStyle = C.tx; g.fillText('R'+r, x, PT+ih+5);
  }

  // 목표선
  if (opt.target != null && opt.target >= b.lo && opt.target <= b.hi){
    const y = Y(opt.target);
    g.strokeStyle = C.target; g.setLineDash([4,3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(PL,y); g.lineTo(w-PR,y); g.stroke(); g.setLineDash([]);
    g.fillStyle = C.target; g.font='8.5px sans-serif'; g.textAlign='left'; g.textBaseline='bottom';
    g.fillText('목표 '+fmtNum(opt.target,opt.dec), PL+4, y-2);
  }

  if (opt.showBase && base.length>1){
    g.strokeStyle = C.base; g.lineWidth = 1.6; g.setLineDash([5,3]);
    g.beginPath(); base.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();
    g.setLineDash([]);
  }

  if (ours.length>1){
    g.beginPath(); ours.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)));
    g.lineTo(X(ours.length-1), PT+ih); g.lineTo(X(0), PT+ih); g.closePath();
    g.fillStyle = C.fill; g.fill();
  }
  g.strokeStyle = C.ours; g.lineWidth = 2.2; g.lineJoin='round';
  g.beginPath(); ours.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();

  const li = ours.length-1;
  if (li >= 0){
    const x = X(li), y = Y(ours[li]);
    g.fillStyle = C.ours; g.beginPath(); g.arc(x,y,3.6,0,7); g.fill();
    g.fillStyle = 'rgba(34,211,238,.22)'; g.beginPath(); g.arc(x,y,7.5,0,7); g.fill();
    const lb = fmtNum(ours[li],opt.dec)+(opt.unit?' '+opt.unit:'');
    g.font='bold 10px Consolas, monospace'; g.textAlign = x>w-70?'right':'left'; g.textBaseline='bottom';
    g.fillStyle=THEME.ink; g.fillText(lb, x + (x>w-70?-8:8), y-6);
  }
}

// 글로벌 손실과 노드별 로컬 손실
function drawLossChart(cv, glob, locals){
  const H = chartH('loss');
  const { g, w, h } = prep(cv, H);
  const PL = 46, PR = 12, PT = 10, PB = 20;
  const iw = w - PL - PR, ih = h - PT - PB;

  const flat = glob.slice();
  Object.keys(locals).forEach(k => flat.push.apply(flat, locals[k]));
  if (flat.length < 2){
    g.fillStyle = C.tx; g.font='11px sans-serif'; g.textAlign='center';
    g.fillText('Federated 라운드가 시작되면 손실 곡선이 표시됩니다', w/2, h/2);
    return;
  }
  const hi = Math.max.apply(null, flat) * 1.08;
  const lo = Math.max(0, Math.min.apply(null, flat) * 0.75);
  const X = i => PL + iw * i / MAX_ROUND;
  const Y = v => PT + ih - ih*(v-lo)/(hi-lo || 1);

  g.font='9px Consolas, monospace'; g.textAlign='right'; g.textBaseline='middle';
  const NY = h < 100 ? 2 : 3;
  for (let i=0;i<=NY;i++){
    const v = lo + (hi-lo)*i/NY, y = Y(v);
    g.strokeStyle=C.grid; g.lineWidth=1;
    g.beginPath(); g.moveTo(PL,y); g.lineTo(w-PR,y); g.stroke();
    g.fillStyle=C.tx; g.fillText(v.toFixed(4), PL-6, y);
  }
  g.textAlign='center'; g.textBaseline='top';
  for (let r=0;r<=MAX_ROUND;r+=6){
    const x = PL + iw*r/MAX_ROUND;
    g.fillStyle=C.tx; g.fillText('R'+r, x, PT+ih+4);
  }

  let ci = 0;
  Object.keys(locals).forEach(k=>{
    const s = locals[k]; const col = C.node[ci++ % C.node.length];
    if (s.length<2) return;
    g.strokeStyle = col; g.globalAlpha = .3; g.lineWidth = 1;
    g.beginPath(); s.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();
    g.globalAlpha = 1;
  });

  g.strokeStyle = THEME.loss; g.lineWidth = 2; g.lineJoin='round';
  g.beginPath(); glob.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();

  const li = glob.length-1;
  if (li>=0){
    g.fillStyle=THEME.loss; g.beginPath(); g.arc(X(li),Y(glob[li]),3.2,0,7); g.fill();
    g.font='bold 9.5px Consolas, monospace'; g.textAlign='right'; g.textBaseline='bottom';
    g.fillText('global '+glob[li].toFixed(4), w-PR, Y(glob[li])-6);
  }
}
