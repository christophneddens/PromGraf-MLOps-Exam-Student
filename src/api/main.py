"""Bike Sharing Predictor API with Prometheus monitoring (exam).

At start-up (only once): download the UCI data, train a RandomForestRegressor
on January 2011 and keep January (with predictions) as Evidently reference.

Endpoints:
- /predict       : predict the number of bikes (cnt) for one hour
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

from fastapi import FastAPI, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import (Counter, Histogram, generate_latest,
                               CollectorRegistry, CONTENT_TYPE_LATEST)

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


@app.get("/metrics")
def metrics():
    """All Prometheus metrics of the API (read by Prometheus)."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
