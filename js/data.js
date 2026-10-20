// 데모에 쓰는 수치와 문구를 모아둔 곳. 화면 문구를 바꾸려면 여기만 보면 된다.

const MAX_ROUND = 24;

// 엣지 플릿. MBS-SBS-UAA-Drone-UE 계층에 Cube Sat 을 더했다.
// tier 는 하드웨어 등급이고 경량화 강도를 정한다.
const NODES = [
  { id:'MBS-01',   nm:'MBS-01',   kind:'MBS',     hw:'L40S 48GB',       tops:362, tier:'hi',  x:380, y:150 },
  { id:'SAT-01',   nm:'SAT-01',   kind:'CubeSat', hw:'RAD-FPGA 2GB',    tops:4,   tier:'xlo', x:672, y:62  },
  { id:'SBS-01',   nm:'SBS-01',   kind:'SBS',     hw:'AGX Orin 64GB',   tops:275, tier:'hi',  x:150, y:252 },
  { id:'SBS-02',   nm:'SBS-02',   kind:'SBS',     hw:'Orin NX 16GB',    tops:100, tier:'mid', x:330, y:252 },
  { id:'SBS-03',   nm:'SBS-03',   kind:'SBS',     hw:'Pi5 + Hailo-8',   tops:26,  tier:'lo',  x:510, y:252 },
  { id:'UAA-01',   nm:'UAA-01',   kind:'UAA',     hw:'Orin Nano 8GB',   tops:40,  tier:'mid', x:66,  y:356 },
  { id:'DRONE-01', nm:'DRONE-01', kind:'Drone',   hw:'Orin Nano 4GB',   tops:20,  tier:'lo',  x:196, y:356 },
  { id:'DRONE-02', nm:'DRONE-02', kind:'Drone',   hw:'Orin Nano 4GB',   tops:20,  tier:'lo',  x:326, y:356 },
  { id:'VEH-01',   nm:'VEH-01',   kind:'UE',      hw:'Snapdragon Ride', tops:30,  tier:'mid', x:456, y:356 },
  // E2 로 중간에 합류하는 저사양 노드
  { id:'DRONE-03', nm:'DRONE-03', kind:'Drone',   hw:'Coral TPU 1GB',   tops:4,   tier:'xlo', x:586, y:356, joinsLater:true }
];

const LINKS = [
  ['CLOUD','MBS-01'], ['MBS-01','SAT-01'],
  ['MBS-01','SBS-01'], ['MBS-01','SBS-02'], ['MBS-01','SBS-03'],
  ['SBS-01','UAA-01'], ['SBS-01','DRONE-01'],
  ['SBS-02','DRONE-02'], ['SBS-02','VEH-01'],
  ['SBS-03','DRONE-03']
];

const CLOUD_POS = { x:380, y:54, w:214, h:36 };

// 모델 생성 파이프라인 단계. 각 단계의 이름과 내용.
const STEPS = [
  { n:'Training Data Collecting',    d:'요청 이력 수집 · 도메인 데이터 프로파일링', ms:900  },
  { n:'자동화 Preprocessing',         d:'결측 보정 · 특징 선택 · 정규화 자동 수행',   ms:900  },
  { n:'Model Architecture Dictionary',d:'Foundation Model Zoo 탐색',              ms:1000 },
  { n:'Model Creation',              d:'Task-conditioned 라우팅 · LoRA 어댑터 합성', ms:1100 },
  { n:'Model Optimization',          d:'노드별 양자화(INT4/8) · 구조적 pruning',    ms:1100 },
  { n:'Trained Model Recommendation',d:'글로벌/로컬 모델 · 손실함수 · 하이퍼파라미터', ms:900  },
  { n:'분산 엣지 배포',                d:'하드웨어 프로파일별 차등 배포',             ms:1200 },
  { n:'Federated LoRA Aggregation',  d:'로컬 PEFT → 어댑터 집계 → 글로벌 갱신',      ms:0    }
];

// 서비스 컴포넌트 3종과 응용 1종.
const TASKS = {

  // 콘텐츠 캐싱
  caching: {
    id:'caching', label:'콘텐츠·모델 캐싱', comp:'③',
    compName:'자율·연계통합 데이터 분석',
    nl:'혼잡 시간대 도심 셀에서 콘텐츠 요청과 엣지 AI 추론 요청이 동시에 몰린다. 각 SBS가 t+1 시점에 어떤 콘텐츠·모델 가중치·KV 캐시를 보유해야 백홀 사용량과 응답 지연이 최소가 되는지 예측하는 모델을 만들어줘.',
    spec:[
      { k:'입력 데이터', v:'요청 이력 14일 · 4.2M rows' },
      { k:'도메인',      v:'Content & Model Caching' },
      { k:'목표 지표',   v:'캐시 히트율 ↑ / 백홀 ↓' },
      { k:'제약',        v:'SBS당 VRAM ≤ 16 GB' }
    ],
    candidates:[
      { n:'TimeFM-Base-340M', m:'시계열 FM · 340M', s:94, pick:true },
      { n:'CacheFormer-45M',  m:'도메인 특화 · 45M', s:87 },
      { n:'CLSTM-24M',        m:'기준선 · 24M', s:71 }
    ],
    loss:'Weighted Focal-MSE (인기도 롱테일 보정)',
    hparams:[
      { k:'LoRA rank', v:'r=16 / α=32' }, { k:'Learning rate', v:'2e-4 cosine' },
      { k:'Batch',     v:'64' },          { k:'Local epoch',   v:'2' },
      { k:'Aggregation',v:'FedAvg + DP ε=4.0' }, { k:'Client 표본', v:'6 / 9' }
    ],
    models:{ hi:'TimeFM-Base-340M', mid:'TimeFM-Small-120M', lo:'TimeFM-Small-120M', xlo:'CacheFormer-45M' },
    baseline:'LRU 캐시 정책',
    vs:{ label:'Inv.MSE', base:0.825, target:0.95 },
    dataGB:2.4, fullGB:1.36, loraMB:4.1,
    kpis:[
      { id:'hit',  l:'캐시 히트율',    u:'%',  from:62.4, to:89.1, base:68.5, dir:'up',   dec:1 },
      { id:'bh',   l:'백홀 사용량',    u:'%',  from:100,  to:59.0, base:94.2, dir:'down', dec:1 },
      { id:'kv',   l:'KV 캐시 재사용', u:'%',  from:31.2, to:76.4, base:34.0, dir:'up',   dec:1 },
      { id:'lat',  l:'E2E 응답 지연',  u:'ms', from:18.4, to:4.2,  base:15.1, dir:'down', dec:1 }
    ],
    mainKpi:'hit', chartTitle:'캐시 히트율 추이 (라운드별)',
    view:'cache'
  },

  // 추론 캐스케이드와 오프로딩
  offloading: {
    id:'offloading', label:'추론 오프로딩', comp:'④',
    compName:'컴퓨팅 자원 관리',
    nl:'엣지 노드에서 처리하지 못하는 추론만 클라우드로 승격하고 싶다. 각 요청의 난이도와 노드 잔여 자원을 보고 엣지에서 끝낼지 클라우드로 올릴지 판단하는 라우팅 모델을 만들어줘.',
    spec:[
      { k:'입력 데이터', v:'태스크 실행 이력 24h · 자원 사용률' },
      { k:'도메인',      v:'Inference Cascade & Offloading' },
      { k:'목표 지표',   v:'P95 지연 ↓ / 엣지 처리율 ↑' },
      { k:'제약',        v:'응답 품질 손실 ≤ 1%p' }
    ],
    candidates:[
      { n:'EdgeLM-1.1B-Router', m:'SLM 라우터 · 1.1B', s:91, pick:true },
      { n:'MoE-Router-8x0.3B',  m:'Mixture-of-Experts', s:84 },
      { n:'MLP-Scheduler',      m:'기준선 · 2M', s:63 }
    ],
    loss:'Cost-sensitive BCE (지연 페널티 가중)',
    hparams:[
      { k:'LoRA rank', v:'r=8 / α=16' },  { k:'Learning rate', v:'1e-4 linear' },
      { k:'Batch',     v:'128' },          { k:'Local epoch',   v:'1' },
      { k:'Aggregation',v:'FedProx μ=0.01' }, { k:'Client 표본', v:'8 / 9' }
    ],
    models:{ hi:'EdgeLM-1.1B-Router', mid:'EdgeLM-1.1B-Router', lo:'EdgeLM-0.5B-Router', xlo:'MLP-Scheduler-2M' },
    baseline:'최근접 노드 고정 할당',
    vs:{ label:'F1', base:0.420, target:0.82 },
    dataGB:5.8, fullGB:2.20, loraMB:2.6,
    kpis:[
      { id:'edge', l:'엣지 처리율',     u:'%',  from:41.0, to:87.2, base:52.4, dir:'up',   dec:1 },
      { id:'p95',  l:'P95 추론 지연',   u:'ms', from:214,  to:38,   base:176,  dir:'down', dec:0 },
      { id:'en',   l:'에너지 소비',     u:'%',  from:100,  to:44.0, base:91.0, dir:'down', dec:1 },
      { id:'qual', l:'응답 품질',       u:'%',  from:94.1, to:97.8, base:94.6, dir:'up',   dec:1 }
    ],
    mainKpi:'edge', chartTitle:'엣지 처리율 추이 (라운드별)',
    view:'cascade'
  },

  // 채널 추정과 자원 할당
  channel: {
    id:'channel', label:'채널추정 · 자원할당', comp:'⑤',
    compName:'네트워크 자원 관리',
    nl:'MBS-SBS-UAA-드론-UE와 LEO 큐브위성이 섞인 6G 비지상망(NTN)에서, 파일럿 신호로부터 t+1 서브캐리어 채널을 추정하고 Tx Power와 대역폭을 동적으로 재할당하는 모델을 만들어줘.',
    spec:[
      { k:'입력 데이터', v:'CQI/CSI History · 전 스펙트럼' },
      { k:'도메인',      v:'Channel Estimation & DRA' },
      { k:'목표 지표',   v:'스펙트럼 효율 ↑ / NMSE ↓' },
      { k:'제약',        v:'추론 지연 ≤ 0.1 ms' }
    ],
    candidates:[
      { n:'ChannelFM-CLSTM-24M', m:'도메인 FM · 24M', s:93, pick:true },
      { n:'Transformer-CE-60M',  m:'Attention 기반 · 60M', s:88 },
      { n:'LS + MMSE',           m:'전통 추정 기법', s:52 }
    ],
    loss:'NMSE + Spectral-Efficiency Regularizer',
    hparams:[
      { k:'LoRA rank', v:'r=32 / α=64' }, { k:'Learning rate', v:'3e-4 cosine' },
      { k:'Batch',     v:'32' },           { k:'Local epoch',   v:'3' },
      { k:'Aggregation',v:'Secure Aggregation' }, { k:'Client 표본', v:'7 / 9' }
    ],
    models:{ hi:'ChannelFM-CLSTM-24M', mid:'ChannelFM-CLSTM-24M', lo:'ChannelFM-Lite-8M', xlo:'ChannelFM-Lite-8M' },
    baseline:'LS 추정 + 고정 자원 할당',
    vs:{ label:'Inv.MSE', base:0.825, target:0.95 },
    dataGB:11.2, fullGB:0.09, loraMB:1.2,
    kpis:[
      { id:'se',   l:'하향 스펙트럼 효율', u:'b/s/Hz', from:22.0,  to:29.4,  base:22.6,  dir:'up',   dec:1 },
      { id:'nmse', l:'채널 추정 NMSE',    u:'dB',     from:-12.4, to:-24.8, base:-13.9, dir:'down', dec:1 },
      { id:'ee',   l:'네트워크 에너지 효율',u:'×',      from:1.0,   to:26.5,  base:1.4,   dir:'up',   dec:1 },
      { id:'ilat', l:'추론 지연',         u:'ms',     from:0.94,  to:0.09,  base:0.88,  dir:'down', dec:2 }
    ],
    mainKpi:'se', chartTitle:'하향 스펙트럼 효율 추이 (목표 30 b/s/Hz)',
    view:'heat'
  },

  // 다양한 응용 서비스. 재난 대응 드론 VLM
  drone: {
    id:'drone', label:'재난대응 VLM', comp:'응용',
    compName:'다양한 응용 서비스',
    nl:'재난 현장에 투입된 드론들이 각자 촬영한 영상에서 요구조자와 위험 요소를 찾고, 확신이 낮은 장면만 클라우드 대형 VLM에 물어보게 해줘. 현장 영상은 절대 외부로 반출하면 안 된다.',
    spec:[
      { k:'입력 데이터', v:'드론 온보드 영상 (반출 금지)' },
      { k:'도메인',      v:'Disaster Response VLM' },
      { k:'목표 지표',   v:'mAP ↑ / 오탐률 ↓' },
      { k:'제약',        v:'드론 VRAM ≤ 4 GB · 30 FPS' }
    ],
    candidates:[
      { n:'EdgeVLM-2B (INT4)',  m:'Vision-Language · 2B', s:92, pick:true },
      { n:'EdgeVLM-Nano-0.5B',  m:'초경량 VLM · 0.5B', s:85 },
      { n:'YOLO-v8n',           m:'탐지 전용 · 3.2M', s:74 }
    ],
    loss:'Focal + Grounding Contrastive',
    hparams:[
      { k:'LoRA rank', v:'r=16 / α=32' }, { k:'Learning rate', v:'1e-4 cosine' },
      { k:'Batch',     v:'16' },           { k:'Local epoch',   v:'2' },
      { k:'Aggregation',v:'FedAvg + Secure Agg' }, { k:'Client 표본', v:'5 / 9' }
    ],
    models:{ hi:'EdgeVLM-2B', mid:'EdgeVLM-2B', lo:'EdgeVLM-Nano-0.5B', xlo:'YOLO-v8n' },
    baseline:'단일 클라우드 VLM 호출',
    vs:{ label:'F1', base:0.420, target:0.82 },
    dataGB:38.6, fullGB:4.10, loraMB:6.8,
    kpis:[
      { id:'map',  l:'요구조자 탐지 mAP@50', u:'%',  from:58.3, to:84.6, base:62.1, dir:'up',   dec:1 },
      { id:'fp',   l:'오탐률',              u:'%',  from:21.4, to:5.8,  base:18.9, dir:'down', dec:1 },
      { id:'rep',  l:'상황보고 정확도',      u:'%',  from:47.2, to:88.3, base:51.0, dir:'up',   dec:1 },
      { id:'dlat', l:'탐지 지연',           u:'ms', from:240,  to:61,   base:205,  dir:'down', dec:0 }
    ],
    mainKpi:'map', chartTitle:'요구조자 탐지 mAP@50 추이 (라운드별)',
    view:'drone'
  }
};

const TASK_ORDER = ['caching','offloading','channel','drone'];

// 진화 이벤트 E1~E3.
const EVENTS = {
  E1:{
    id:'E1', name:'분포 급변 → Drift 탐지 → 어댑터 교체',
    node:'SBS-02',
    log:[
      ['warn','SBS-02',     '입력 분포 급변 감지 · PSI = 0.34 (임계 0.20)'],
      ['err', 'Statistical','예측 손실 급등 : 0.0041 → 0.0193 (4.7×)'],
      ['evt', 'Replacement','Model Replacement Decision : DRIFT DETECTED'],
      ['i',   'Model Zoo',  '대체 아키텍처 재탐색 · 후보 3종 재평가'],
      ['ok',  'Replacement','LoRA 어댑터만 교체 (전체 재학습 4.2h → 3.5min)'],
      ['ok',  'SBS-02',     '진화 완료 · 재수렴 대기']
    ],
    penalty:0.42, recover:3
  },
  E2:{
    id:'E2', name:'신규 저사양 노드 합류 → 자동 양자화 재추천',
    node:'DRONE-03',
    log:[
      ['evt','Fleet',      'DRONE-03 합류 요청 · 하드웨어 프로파일 수신'],
      ['i',  'Profiler',   'Coral TPU 1GB / 4 TOPS, 현재 모델 배포 불가'],
      ['i',  'Optimization','구조적 pruning 62% + INT4 양자화 자동 적용'],
      ['ok', 'Recommend',  '경량 모델 재추천 · LoRA r=4 로 하향'],
      ['ok', 'DRONE-03',   '배포 완료 · 이종 엣지 플릿에 편입']
    ],
    penalty:0.12, recover:2
  },
  E3:{
    id:'E3', name:'캐스케이드 폭주 → 자율 재증류',
    node:'SBS-03',
    log:[
      ['warn','Cascade',   '클라우드 승격률 13% → 41% 급등'],
      ['err', 'KPI',       '지연·에너지·탄소 KPI 동시 악화 감지'],
      ['i',   'Diagnosis', '엣지 student 모델 열화가 원인으로 판정'],
      ['evt', 'Distill',   'Cloud teacher → Edge student 재증류 트리거'],
      ['ok',  'Distill',   '지식 증류 완료 · 승격률 정상 범위 복귀']
    ],
    penalty:0.55, recover:3
  }
};

const QUANT = { hi:'FP16', mid:'INT8', lo:'INT4', xlo:'INT4' };
const RANK  = { hi:16, mid:16, lo:8, xlo:4 };
const TIER_KO = { hi:'고사양', mid:'중사양', lo:'저사양', xlo:'초저사양' };
