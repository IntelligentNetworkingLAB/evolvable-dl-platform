// canvas, svg 로 직접 그리는 부분(차트, 히트맵)의 색.
// CSS 쪽은 style.css 의 :root / [data-theme=...] 에 같은 이름으로 있다.
//
//   navy      남색 계열. 기본값
//   paper     무채색, 강조 한 색
//   slate     저채도 다크
//   contrast  고대비

const THEMES = {

  navy: {
    ink:'#16202e', ink2:'#55637a', ink3:'#8b98ab',
    grid:'#e9edf3', panel:'#ffffff',
    ours:'#1f5fa8', base:'#a4aebd', target:'#15754f',
    fill:'rgba(31,95,168,0.09)', loss:'#96690f',
    blue:'#1f5fa8', teal:'#0f6f86', green:'#15754f',
    amber:'#96690f', red:'#ad3a30', violet:'#574c8c',
    series:['#1f5fa8','#574c8c','#0f6f86','#96690f','#15754f',
            '#7d8aa0','#ad3a30','#4a6f8a','#6b5b8e','#3f7a6a'],
    heatLo:[228,233,240], heatHi:[21,117,79]
  },

  paper: {
    ink:'#1c1c1a', ink2:'#585854', ink3:'#8d8d88',
    grid:'#ebebe8', panel:'#ffffff',
    ours:'#3a5a80', base:'#b0b0aa', target:'#4a6b52',
    fill:'rgba(58,90,128,0.08)', loss:'#7d6420',
    blue:'#3a5a80', teal:'#4a6b73', green:'#4a6b52',
    amber:'#7d6420', red:'#8f4038', violet:'#56506e',
    series:['#3a5a80','#56506e','#4a6b73','#7d6420','#4a6b52',
            '#8d8d88','#8f4038','#5c6f7d','#6b6480','#5a7a68'],
    heatLo:[233,233,229], heatHi:[74,107,82]
  },

  slate: {
    ink:'#e3e8ef', ink2:'#a3adbb', ink3:'#6f7b8a',
    grid:'#28313f', panel:'#1b212b',
    ours:'#6fa4dd', base:'#5c6878', target:'#5db98a',
    fill:'rgba(111,164,221,0.12)', loss:'#d0a349',
    blue:'#6fa4dd', teal:'#58a8bd', green:'#5db98a',
    amber:'#d0a349', red:'#dd7a70', violet:'#9b8fd4',
    series:['#6fa4dd','#9b8fd4','#58a8bd','#d0a349','#5db98a',
            '#8b98ab','#dd7a70','#7fa3c4','#a396cd','#6fb3a0'],
    heatLo:[38,46,58], heatHi:[93,185,138]
  },

  contrast: {
    ink:'#000000', ink2:'#3a3a3a', ink3:'#666666',
    grid:'#dcdcdc', panel:'#ffffff',
    ours:'#0b4d99', base:'#8c8c8c', target:'#0a6b43',
    fill:'rgba(11,77,153,0.10)', loss:'#7a5200',
    blue:'#0b4d99', teal:'#0a5f73', green:'#0a6b43',
    amber:'#7a5200', red:'#a01b12', violet:'#443a80',
    series:['#0b4d99','#443a80','#0a5f73','#7a5200','#0a6b43',
            '#555555','#a01b12','#26608c','#5a4a99','#12775a'],
    heatLo:[226,226,226], heatHi:[10,107,67]
  }
};

const THEME_LIST = [
  ['navy',     'A · 네이비 라이트'],
  ['paper',    'B · 무채색 인쇄물'],
  ['slate',    'C · 슬레이트 다크'],
  ['contrast', 'D · 고대비']
];

// charts.js, topology.js, ui.js 가 이 이름을 그대로 참조한다
let THEME = THEMES.navy;
let THEME_NAME = 'navy';

function setTheme(name){
  if (!THEMES[name]) name = 'navy';
  THEME = THEMES[name];
  THEME_NAME = name;
  document.documentElement.setAttribute('data-theme', name);
  try { localStorage.setItem('edl-theme', name); } catch (e) {}
  return name;
}

function savedTheme(){
  try { return localStorage.getItem('edl-theme') || 'navy'; }
  catch (e) { return 'navy'; }
}
