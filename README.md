# Evolvable Deep Learning Model Generation Platform for Edge Computing

분산 엣지에서 연합학습으로 도메인 특화 모델을 만들고, 그 모델이 응용 지표를 얼마나 끌어올리는지 한 화면에서 보여준다.
응용은 콘텐츠 캐싱, 추론 오프로딩, 채널 추정 세 가지이고, 라운드 중에 데이터 분포 급변·노드 합류·연쇄 폭주 같은
이벤트를 일으켜 볼 수 있다.

백엔드는 numpy 로 LoRA 어댑터 연합학습(FedAvg, 등급별 양자화·프루닝)을 돌린다. 데이터는 합성이다.

## 실행

```bash
pip install -r requirements.txt
python backend/serve.py            # http://127.0.0.1:8000
```

`--host`, `--port` 로 바꿀 수 있다 (환경변수 `EDL_HOST`, `EDL_PORT` 도 된다).

서버 없이 `index.html` 을 브라우저로 열어도 된다. 이때는 `data/runs/` 에 미리 돌려 둔 학습 기록을 재생하고, 기록도
없으면 시뮬레이션으로 동작한다. 화면 오른쪽 위 배지가 지금 어느 모드인지 알려준다
(`실측 · 실시간` / `실측 기록` / `시뮬레이션`).

화면 사용: 위쪽 탭에서 응용을 고르고 `시연 시작`을 누른다. 라운드가 진행되는 동안 `E1`, `E2`, `E3` 버튼으로 이벤트를 일으킨다.

## 데이터와 기록 다시 만들기

```bash
python backend/make_data.py          # data/*.csv 표본
python backend/run_experiment.py     # data/runs/*.js 학습 기록 (--rounds, --seed, --no-events)
```

## 구성

```
index.html, css/, js/     프론트엔드 (바닐라 JS, 외부 라이브러리 없음)
backend/serve.py          HTTP 서버 (표준 라이브러리) + 정적 파일
backend/edl/              LoRA-MLP, FedAvg, 데이터 생성, 응용별 태스크
backend/tests/            pytest
data/                     표본 CSV, 학습 기록
```

## 테스트

```bash
pip install -r requirements-dev.txt
cd backend && python -m pytest
```

## 참고

- 재난 대응 VLM 탭은 시뮬레이션이다.
- 채널 추정 탭은 이벤트가 걸린 기본 기록에서 LS 기준선보다 낮게 끝난다. 이벤트 없이 다시 기록하려면 `run_experiment.py --no-events`.

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)),(No.2019-0-01287, Evolvable Deep Learning Model Generation Platform for Edge Computing)

## 라이선스

Apache-2.0
