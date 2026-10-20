# Evolvable Deep Learning Model Generation Platform for Edge Computing

Builds domain-specific models on distributed edge nodes with federated learning and shows how much the models improve application-level metrics. Three applications are included: content caching, inference offloading, and channel estimation. Events such as data distribution shift, node join, and cascade overload can be triggered while training rounds are running.

The backend runs federated learning of LoRA adapters (FedAvg with per-tier quantization and pruning) in plain numpy.

## Getting started

```bash
pip install -r requirements.txt
python backend/serve.py            # http://127.0.0.1:8000
```

Use `--host` / `--port` (or the `EDL_HOST` / `EDL_PORT` environment variables) to change the address.

`index.html` can also be opened directly in a browser without the server. In that case it replays the training records in `data/runs/`, or falls back to a simulation if none are found. The badge at the top right shows the current mode (`실측 · 실시간`, `실측 기록`, or `시뮬레이션`).

## Regenerating data and training records

```bash
python backend/make_data.py          # sample CSVs in data/
python backend/run_experiment.py     # training records in data/runs/ (--rounds, --seed, --no-events)
```

## Project layout

```
index.html, css/, js/     frontend (vanilla JS, no external libraries)
backend/serve.py          HTTP server (standard library) and static files
backend/edl/              LoRA-MLP, FedAvg, data generation, per-application tasks
backend/tests/            pytest suite
data/                     sample CSVs and training records
```

## Tests

```bash
pip install -r requirements-dev.txt
cd backend && python -m pytest
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)),(No.2019-0-01287, Evolvable Deep Learning Model Generation Platform for Edge Computing)

## License

Apache-2.0
