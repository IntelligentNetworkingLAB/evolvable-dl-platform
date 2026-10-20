// 엣지 토폴로지 SVG 와 패킷 이동.
// 계층은 CLOUD > MBS/CubeSat > SBS > UAA/Drone/UE.
//
// 애니메이션을 rAF 가 아니라 setInterval 로 돌리는 이유:
// 창이 가려지거나 원격 데스크톱이면 rAF 가 초당 1프레임까지 떨어진다.
// 그러면 패킷이 도착을 못 하고 flySet 의 Promise 가 안 풀려서 시연이 멈춘다.
// 같은 이유로 flySet 에 워치독을 뒀다.

const Topo = (function(){
  const NS = 'http://www.w3.org/2000/svg';
  const BW = 82, BH = 44;              // 노드 박스 크기
  const FPS_MS = 33;                   // 애니메이션 틱 간격
  let svg, gLink, gNode, gPkt;
  const elNode = {}, elLink = {}, elModel = {};
  let parent = {}, pkts = [], timer = null, last = 0;

  function mk(tag, attrs, cls){
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (cls) e.setAttribute('class', cls);
    return e;
  }
  function txt(x, y, s, cls){
    const t = mk('text', { x:x, y:y }, cls); t.textContent = s; return t;
  }
  function pos(id){
    if (id === 'CLOUD') return { x:CLOUD_POS.x, y:CLOUD_POS.y };
    const n = NODES.find(n => n.id === id);
    return n ? { x:n.x, y:n.y } : { x:0, y:0 };
  }

  function init(){
    svg = document.getElementById('topo');
    stopAll();
    svg.innerHTML = '';

    parent = {};
    LINKS.forEach(l => parent[l[1]] = l[0]);

    const gT = mk('g');
    [['CLOUD', 54], ['NTN / MACRO', 150],
     ['SMALL CELL', 252], ['UE / MOBILE EDGE', 356]]
      .forEach(p => gT.appendChild(txt(10, p[1]+3, p[0], 'tp-tier')));
    svg.appendChild(gT);

    gLink = mk('g'); svg.appendChild(gLink);
    gNode = mk('g'); svg.appendChild(gNode);
    gPkt  = mk('g'); svg.appendChild(gPkt);

    LINKS.forEach(l => {
      const a = l[0], b = l[1], A = pos(a), B = pos(b);
      const ay = a === 'CLOUD' ? A.y + CLOUD_POS.h/2 : A.y + BH/2;
      const by = B.y - BH/2;
      const ln = mk('path', {
        d:'M'+A.x+','+ay+' C'+A.x+','+((ay+by)/2)+' '+B.x+','+((ay+by)/2)+' '+B.x+','+by
      }, 'tp-link');
      elLink[a+'>'+b] = ln;
      gLink.appendChild(ln);
    });

    const gc = mk('g', { id:'cloudBox' });
    gc.appendChild(mk('rect', {
      x:CLOUD_POS.x-CLOUD_POS.w/2, y:CLOUD_POS.y-CLOUD_POS.h/2,
      width:CLOUD_POS.w, height:CLOUD_POS.h, rx:4
    }, 'tp-cloud'));
    gc.appendChild(txt(CLOUD_POS.x, CLOUD_POS.y+1, 'Cloud · Model Aggregator', 'tp-cloudtx'));
    gc.appendChild(txt(CLOUD_POS.x, CLOUD_POS.y+13, 'Autonomous Model Aggregation', 'tp-hw'));
    gNode.appendChild(gc);

    NODES.forEach(n => {
      const g = mk('g', { transform:'translate('+n.x+','+n.y+')' }, 'tp-node st-idle');
      g.appendChild(mk('rect', { x:-BW/2, y:-BH/2, width:BW, height:BH, rx:4 }, 'tp-box'));
      g.appendChild(txt(0, -12, n.kind.toUpperCase(), 'tp-kind'));
      g.appendChild(txt(0, 3,  n.nm, 'tp-nm'));
      g.appendChild(txt(0, 14, n.hw, 'tp-hw'));
      const md = txt(0, BH/2 + 11, '—', 'tp-md');
      md.setAttribute('fill', THEME.ink3);
      g.appendChild(md);
      elModel[n.id] = md;
      elNode[n.id] = g;
      gNode.appendChild(g);
    });
  }

  function setState(id, st){
    const g = elNode[id]; if (!g) return;
    g.setAttribute('class', 'tp-node st-' + st);
  }
  function isVisible(id){
    const g = elNode[id]; return !!g && g.style.display !== 'none';
  }
  function setModel(id, label, quant){
    const m = elModel[id]; if (!m) return;
    m.textContent = label ? label + ' · ' + quant : '—';
    m.setAttribute('fill', !label ? THEME.ink3
      : quant === 'INT4' ? THEME.amber : quant === 'INT8' ? THEME.teal : THEME.blue);
  }
  function reveal(id){
    const g = elNode[id]; if (!g) return;
    g.style.display = '';
    const ln = elLink['SBS-03>' + id]; if (ln) ln.style.display = '';
    setState(id, 'new');
  }
  function hideLate(){
    NODES.filter(n => n.joinsLater).forEach(n => {
      if (elNode[n.id]) elNode[n.id].style.display = 'none';
      const ln = elLink['SBS-03>' + n.id]; if (ln) ln.style.display = 'none';
    });
  }
  function litPath(id, on){
    let c = id;
    while (parent[c]){
      const ln = elLink[parent[c] + '>' + c];
      if (ln) ln.classList.toggle('act', on);
      c = parent[c];
    }
  }
  function unlitAll(){
    Object.keys(elLink).forEach(k => elLink[k].classList.remove('act'));
  }

  function route(id){
    const way = [pos(id)];
    let c = id, guard = 0;
    while (parent[c] && guard++ < 12) { way.unshift(pos(parent[c])); c = parent[c]; }
    return way;
  }

  function fly(id, dir, onDone){
    if (!isVisible(id)) { if (onDone) onDone(); return; }
    const way = route(id);
    const pts = dir === 'up' ? way.slice().reverse() : way;
    const el = mk('circle', { r:3.4, cx:pts[0].x, cy:pts[0].y }, 'pkt ' + dir);
    gPkt.appendChild(el);
    pkts.push({ el:el, pts:pts, seg:0, t:0, onDone:onDone });
    litPath(id, true);
    start();
  }

  // ids 가 다 도착하면 allDone. 워치독 때문에 최소 한 번은 불린다
  function flySet(ids, dir, allDone){
    let left = ids.length, fired = false;
    const finish = () => {
      if (fired) return; fired = true;
      if (allDone) allDone();
    };
    if (!left) { finish(); return; }
    ids.forEach((id, i) => setTimeout(() => {
      fly(id, dir, () => { if (--left <= 0) finish(); });
    }, i * 110));
    // 애니메이션이 멈춰도 시연은 계속 굴러가야 한다
    setTimeout(() => {
      if (!fired){ flushAll(); finish(); }
    }, ids.length * 110 + 3200);
  }

  function flushAll(){
    pkts.forEach(p => { if (p.el.parentNode) p.el.remove(); });
    pkts = [];
    stopAll(); unlitAll();
  }
  function stopAll(){
    if (timer !== null){ clearInterval(timer); timer = null; }
  }
  function start(){
    if (timer !== null) return;
    last = performance.now();
    timer = setInterval(tick, FPS_MS);
  }

  function tick(){
    const now = performance.now();
    const dt = Math.min(120, now - last);
    last = now;
    for (let i = pkts.length - 1; i >= 0; i--){
      const p = pkts[i];
      const a = p.pts[p.seg], b = p.pts[p.seg+1];
      if (!b){
        if (p.el.parentNode) p.el.remove();
        pkts.splice(i,1);
        if (p.onDone) p.onDone();
        continue;
      }
      const len = Math.hypot(b.x-a.x, b.y-a.y) || 1;
      p.t += dt * 0.52 / len;   // ≈ 520 px/s
      if (p.t >= 1){ p.t = 0; p.seg++; continue; }
      p.el.setAttribute('cx', a.x + (b.x-a.x)*p.t);
      p.el.setAttribute('cy', a.y + (b.y-a.y)*p.t);
    }
    if (!pkts.length){ stopAll(); unlitAll(); }
  }

  // 집계 순간 표시. 글로우 대신 테두리를 굵게 한다
  function pulseCloud(){
    const r = svg && svg.querySelector('.tp-cloud'); if (!r) return;
    r.style.transition = 'none';
    r.style.stroke = THEME.blue;
    r.style.strokeWidth = '2.2';
    r.style.fill = THEME['blue-lt'] || '#eaf1fa';
    setTimeout(() => {
      r.style.transition = 'stroke-width .5s, fill .5s';
      r.style.stroke = ''; r.style.strokeWidth = ''; r.style.fill = '';
    }, 300);
  }

  return { init, setState, setModel, reveal, hideLate, flySet,
           pulseCloud, isVisible, flushAll };
})();
