"""Bike Sharing Predictor API with Prometheus monitoring (exam).

At start-up (only once): download the UCI data, train a RandomForestRegressor
on January 2011 and keep January (with predictions) as Evidently reference.

Endpoints:
- /predict       : predict the number of bikes (cnt) for one hour
- /evaluate      : predict a period, compare it with January (Evidently),
                   update the Prometheus gauges
- /trigger-drift : evaluate a deliberately distorted batch (used by make fire-alert)
- /metrics       : all metrics in Prometheus format
"""
import datetime
import io
import logging
import time
import warnings
import zipfile
from typing import Any, Optional

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, MAPE, RMSE, R2Score
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import (Counter, Histogram, generate_latest,
                               CollectorRegistry, Gauge, CONTENT_TYPE_LATEST)

# Hide library warnings so that the logs stay readable
warnings.filterwarnings("ignore")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---
# Own registry: /metrics shows only our metrics
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "api_requests_total", "Total number of API requests",
    ["endpoint", "method", "status_code"], registry=registry)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "API request duration in seconds",
    ["endpoint", "method", "status_code"], registry=registry)

# Model quality on the last evaluated period (set by /evaluate)
RMSE_GAUGE = Gauge("model_rmse_score", "RMSE of the model on the last evaluated period",
                   registry=registry)
MAE_GAUGE = Gauge("model_mae_score", "MAE of the model on the last evaluated period",
                  registry=registry)
R2_GAUGE = Gauge("model_r2_score", "R2 score of the model on the last evaluated period",
                 registry=registry)

# Metric of my choice: 1 = Evidently detects data drift, 0 = no drift.
# Why it is relevant: RMSE, MAE and R2 need the true number of bikes, which
# is only known afterwards. Drift in the inputs (weather, hour, day ...)
# is an early warning: the model receives data unlike January, the data it
# was trained on, so its predictions may become unreliable.
DRIFT_GAUGE = Gauge("evidently_data_drift_detected_status",
                    "1 if Evidently detects data drift in the last evaluated period, else 0",
                    registry=registry)


# --- Global Variables for Model and Data ---
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DTEDAY_COL_NAME = 'dteday'
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
FEATURES = NUM_FEATS + CAT_FEATS


# --- Data Ingestion and Preparation Functions ---
# (same as in src/evaluation/run_evaluation.py)
def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    logger.info("Fetching data from UCI archive...")
    content = requests.get(DATASET_URL, verify=False, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open("hour.csv"), header=0, sep=',',
                         parse_dates=[DTEDAY_COL_NAME])
    return df


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Sets a DatetimeIndex (date + hour) and sorts the rows by time."""
    raw_data['hr'] = raw_data['hr'].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row[DTEDAY_COL_NAME].date(),
                                              datetime.time(row.hr)),
        axis=1
    )
    return raw_data.sort_index()


def _train_and_predict_reference_model(data: pd.DataFrame):
    """Trains the model on January 2011.

    Returns the model and the January data with a 'prediction' column,
    which Evidently uses as reference.
    """
    reference = data.loc['2011-01-01 00:00:00':'2011-01-31 23:00:00'].copy()
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(reference[FEATURES], reference[TARGET])
    reference[PREDICTION] = model.predict(reference[FEATURES])
    logger.info(f"Model trained on {len(reference)} rows of January 2011.")
    return model, reference


# The model is trained only once, when the API container starts
MODEL, REFERENCE_DATA = _train_and_predict_reference_model(_process_data(_fetch_data()))


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int


# --- Request monitoring (every request except /metrics) ---
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Counts each request and measures how long it took."""
    start = time.time()
    response = await call_next(request)
    if request.url.path != "/metrics":
        labels = [request.url.path, request.method, str(response.status_code)]
        REQUEST_COUNT.labels(*labels).inc()
        REQUEST_LATENCY.labels(*labels).observe(time.time() - start)
    return response


# --- Evaluation logic (used by /evaluate and /trigger-drift) ---
def _make_dataset(df: pd.DataFrame) -> Dataset:
    """Tells Evidently which column is what."""
    definition = DataDefinition(
        numerical_columns=NUM_FEATS + [TARGET, PREDICTION],
        categorical_columns=CAT_FEATS,
        regression=[Regression(target=TARGET, prediction=PREDICTION)],
    )
    columns = FEATURES + [TARGET, PREDICTION]   # dteday is not needed
    return Dataset.from_pandas(df[columns], data_definition=definition)


def _evaluate(current: pd.DataFrame, period_name: str) -> EvaluationReportOutput:
    """Predicts the period, runs the Evidently report, updates the gauges."""
    current = current.copy()
    current[PREDICTION] = MODEL.predict(current[FEATURES])

    report = Report(metrics=[RMSE(), MAE(), MAPE(), R2Score(),
                             DataDriftPreset(columns=FEATURES)])
    snapshot = report.run(current_data=_make_dataset(current),
                          reference_data=_make_dataset(REFERENCE_DATA))

    # The results come in the same order as the metrics above
    results = snapshot.dict()["metrics"]
    rmse = results[0]["value"]
    mae = results[1]["value"]["mean"]
    mape = results[2]["value"]["mean"]
    r2 = results[3]["value"]
    drift_share = results[4]["value"]["share"]   # DriftedColumnsCount
    drift_detected = 1 if drift_share >= 0.5 else 0   # Evidently's default rule

    RMSE_GAUGE.set(rmse)
    MAE_GAUGE.set(mae)
    R2_GAUGE.set(r2)
    DRIFT_GAUGE.set(drift_detected)
    logger.info(f"{period_name}: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.3f} "
                f"drifted columns={drift_share:.0%}")

    return EvaluationReportOutput(
        message=f"Evaluation of '{period_name}' done, Prometheus metrics updated.",
        rmse=rmse, mape=mape, mae=mae, r2score=r2,
        drift_detected=drift_detected, evaluated_items=len(current))


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}


@app.post("/predict", response_model=PredictionOutput)
def predict(item: BikeSharingInput):
    """Predicts the number of bikes for one hour."""
    features = pd.DataFrame([item.model_dump()])[FEATURES]
    prediction = MODEL.predict(features)[0]
    return PredictionOutput(predicted_count=round(float(prediction), 2))


@app.post("/evaluate", response_model=EvaluationReportOutput)
def evaluate(payload: EvaluationData):
    """Evaluates the model on a period sent by run_evaluation.py."""
    current = pd.DataFrame(payload.data)
    missing = [col for col in FEATURES + [TARGET] if col not in current.columns]
    if current.empty or missing:
        raise HTTPException(status_code=400,
                            detail=f"Empty data or missing columns: {missing}")
    return _evaluate(current, payload.evaluation_period_name)


@app.post("/trigger-drift", response_model=EvaluationReportOutput)
def trigger_drift():
    """Simulates a strong drift to test the alert (make fire-alert).

    January data is changed: much warmer, wetter and windier, heavy rain,
    holidays, and five times more bikes than in reality. The RMSE becomes
    very high and Evidently detects data drift.
    """
    current = REFERENCE_DATA.sample(n=200, random_state=42).drop(columns=[PREDICTION])
    for col in ['temp', 'atemp', 'hum', 'windspeed']:
        current[col] = current[col] + 0.5
    current['weathersit'] = 4
    current['holiday'] = 1
    current[TARGET] = current[TARGET] * 5
    return _evaluate(current, "simulated_drift")


@app.get("/metrics")
def metrics():
    """All Prometheus metrics of the API (read by Prometheus)."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
