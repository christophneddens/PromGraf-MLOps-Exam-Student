#bike Sharing — monitoring with Prometheus and Grafana (exam)

A FastAPI service predicts the number of rented bikes per hour (`cnt`, UCI Bike Sharing dataset).
Prometheus collects metrics about the API, the model and the host; Grafana shows them on three
dashboards and raises alerts. Everything — data source, dashboards and the Grafana alert — is
created from files at start-up, so the project runs on any machine with one command: `make`.

##Quick start

```bash
make              # same as "make all": builds and starts bike-api, Prometheus, Grafana, node-exporter
make evaluation   # waits for the API, then sends the three February weeks to /evaluate
make traffic      # sends 100 requests to /predict (every 10th one is invalid on purpose -> 422)
make fire-alert   # fires the Grafana alert "High model RMSE" (see "Alerts" below)
make stop         # stops all services
```

At start-up the API downloads the data and trains the model (about 30 s). `make evaluation`
waits for this automatically.

| Service | URL | Notes |
|---|---|---|
| bike-api | http://localhost:8080 | `/predict`, `/evaluate`, `/trigger-drift`, `/metrics`, `/docs` |
| Prometheus | http://localhost:9090 | Status → Target health, Alerts |
| Grafana | http://localhost:3000 | login `admin` / `admin`; Dashboards and Alerting → Alert rules |
| node-exporter | http://localhost:9100/metrics | metrics of the host |

##project structure

```
├── Makefile                         all, stop, evaluation, traffic, fire-alert
├── docker-compose.yml               bike-api, evaluation, prometheus, node-exporter, grafana
├── scripts/generate_traffic.sh      traffic on /predict
├── src/
│   ├── api/                         main.py, Dockerfile, requirements.txt
│   └── evaluation/                  run_evaluation.py (provided), Dockerfile, requirements.txt
└── deployment/
    ├── prometheus/
    │   ├── prometheus.yml           scrapes bike-api, node-exporter and Prometheus itself
    │   └── rules/alert_rules.yml    Prometheus alert BikeApiDown
    └── grafana/
        ├── provisioning/
        │   ├── datasources.yaml     Prometheus data source (fixed uid "prometheus")
        │   ├── dashboards.yaml      loads every JSON file from deployment/grafana/dashboards
        │   └── alerting.yaml        Grafana alert "High model RMSE" (created in the UI, exported)
        └── dashboards/              API-dashboard.json, Model-dashboard.json, Infra-dashboard.json
```

##how it works

1. **Model.** When the `bike-api` container starts, `_fetch_data` and `_process_data` load the UCI
   data and `_train_and_predict_reference_model` trains a `RandomForestRegressor` (50 trees,
   `random_state=42`) once, on January 2011 (688 hours). January with its predictions is kept as the
   Evidently reference.
2. **Metrics** (own `CollectorRegistry`, exposed on `/metrics`):
   - `api_requests_total` (Counter) and `api_request_duration_seconds` (Histogram), labels
     `endpoint`, `method`, `status_code` — recorded for every request by a FastAPI middleware;
   - `model_rmse_score`, `model_mae_score`, `model_r2_score` (Gauges), set by `/evaluate`;
   - `evidently_data_drift_detected_status` (Gauge) — the metric of my choice (see below).
3. **`/evaluate`** receives one period from `run_evaluation.py`, predicts it, and runs an Evidently
   `Report` with `RMSE`, `MAE`, `MAPE`, `R2Score` and `DataDriftPreset` (the 11 features) against the
   January reference. The values are read from `snapshot.dict()` and written to the gauges.
4. **Prometheus** scrapes every 15 s and evaluates the rule file.
5. **Grafana** is provisioned from files: data source, three dashboards and the alert rule.
   Tested on a clean start (`docker compose down -v`, then `make`): everything comes back.

## Metric of my choice: `evidently_data_drift_detected_status`

1 when Evidently finds data drift in the evaluated period, 0 otherwise (Evidently's default rule:
at least half of the 11 input columns have drifted compared with January).
**Why it is relevant:** RMSE, MAE and R² need the true number of rented bikes, which is only known
afterwards. Drift in the inputs (weather, hour, day …) is visible at once. It is an early warning
that the model receives data unlike the data it was trained on, so its predictions may become
unreliable — and a signal to check the data and consider retraining.

##dashboards

| Dashboard | Panels |
|---|---|
| API Performance | request rate by endpoint, error rate (share of non-2xx), latency P95 by endpoint, average latency, total requests |
| Model Performance & Drift | RMSE, MAE, R² and drift status (last value and over time) |
| Infrastructure Overview | CPU used, memory used, disk used (root filesystem), network traffic |

##alerts

| Alert | Defined in | Condition | How to test |
|---|---|---|---|
| `BikeApiDown` | Prometheus, `deployment/prometheus/rules/alert_rules.yml` | `up{job="bike-api"} == 0` for 1 min | `docker compose stop bike-api` |
| `High model RMSE` | Grafana — created in the UI, exported to `deployment/grafana/provisioning/alerting.yaml` | last `model_rmse_score` above 100 for 30 s | `make fire-alert` |

**Which alert does `make fire-alert` test?** The Grafana alert **High model RMSE**, which is based on
an ML metric. `make fire-alert` calls `/trigger-drift`: 200 January rows are deliberately distorted
(temperature, felt temperature, humidity and wind +0.5, heavy rain, holidays, five times more bikes)
and evaluated like a normal period. In my run: RMSE **305.75**, MAE 215.40, R² −0.507, drift detected.
The alert turns *Pending* and, after 30 s, *Firing* (Grafana → Alerting → Alert rules).
`make evaluation` brings the values back to normal and the alert resolves.
(The contact point is Grafana's default e-mail contact point; no mail server is configured, so the
alert is shown in Grafana but no e-mail is sent.)

##results of `make evaluation`

| Period (as defined in `run_evaluation.py`) | Rows | RMSE | MAE | R² | Drifted columns |
|---|---|---|---|---|---|
| week1_february (29 Jan – 7 Feb) | 235 | 17.69 | 10.54 | 0.883 | 64 % → drift |
| week2_february (8 – 14 Feb) | 163 | 20.41 | 13.98 | 0.880 | 73 % → drift |
| week3_february (15 – 21 Feb) | 165 | 37.19 | 24.29 | 0.735 | 64 % → drift |

The error roughly doubles in week 3: days become warmer and more bikes are rented than in any hour
of January. A random forest cannot predict more than the largest values it saw in training, so it
under-predicts busy hours. Drift is detected in every week: the weather columns change, and `mnth`
is always 2 instead of 1. Because the reference is one month only, calendar columns always drift;
a longer reference period (or leaving `mnth` out of the drift check) would make the drift status
more selective.

##notes and choices

- `docker compose` (Compose v2) is used; the starter Makefile called the old `docker-compose`.
- `make all` starts the four services only. The `evaluation` container is started by
  `make evaluation` once the API answers — otherwise it would run before the model is trained.
- `run_evaluation.py` (provided) was changed to evaluate all three weeks, with a 20 s pause so that
  Prometheus records each result.
- Pinned versions: `python:3.12-slim`, `prom/prometheus:v3.5.0`, `prom/node-exporter:v1.9.1`,
  `grafana/grafana:11.4.0`; Evidently 0.7.2 with pandas 2.3.3.
- The network panel shows the traffic seen by the node-exporter container (course set-up, no host
  network mode).

Author: Christoph Neddens — Liora ML Engineering, Prometheus & Grafana module.
