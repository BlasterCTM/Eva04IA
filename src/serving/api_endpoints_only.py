"""Versión mínima de la API: solo endpoints JSON, sin UI o páginas HTML.

Endpoints disponibles:
- GET / -> lista de endpoints
- GET /health
- POST /predict
- GET /model/version
- GET /reports/detail
- GET /reports/season
- GET /plots
- POST /retrain

Uso: `uvicorn src.serving.api_endpoints_only:app --port 8000`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import sys
import json
import joblib
import logging
import subprocess
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Response
from pydantic import BaseModel

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    METRICS_ENABLED = True
except Exception:
    # If prometheus_client is not available in the runtime, disable metrics gracefully
    METRICS_ENABLED = False


from src.ingenieria_caracteristicas import agregar_caracteristicas


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/modelo_demanda_v1.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "models/metadata.json"))
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Logging
logger = logging.getLogger("api")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Normalize LOG_LEVEL (case-insensitive) and apply safe fallback
_level_raw = os.getenv("LOG_LEVEL", "INFO")
try:
    _level_name = str(_level_raw).upper()
    _level_value = logging.getLevelName(_level_name)
    if isinstance(_level_value, int):
        logger.setLevel(_level_value)
    else:
        logger.setLevel(logging.INFO)
except Exception:
    logger.setLevel(logging.INFO)

# Prometheus metrics (if available)
if METRICS_ENABLED:
    REQUEST_COUNT = Counter("api_request_count", "Total API requests", ["method", "endpoint"])
    PREDICT_TIME = Histogram("api_predict_seconds", "Time spent in predict")
else:
    class _NoopMetric:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            return None

        def time(self):
            class _Ctx:
                def __enter__(self_self):
                    return None

                def __exit__(self_self, exc_type, exc, tb):
                    return False

            return _Ctx()

    REQUEST_COUNT = _NoopMetric()
    PREDICT_TIME = _NoopMetric()


def load_model() -> Any | None:
    if MODEL_PATH.exists():
        try:
            logger.info(f"Cargando modelo desde {MODEL_PATH}")
            return joblib.load(MODEL_PATH)
        except Exception:
            logger.exception("Error cargando modelo")
            return None
    logger.warning(f"Modelo no encontrado en {MODEL_PATH}")
    return None


MODEL = load_model()


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Modelo Demanda - Minimal API",
        "endpoints": [
            "/health",
            "/predict (POST)",
            "/model/version",
            "/reports/detail?rows=N",
            "/reports/season",
            "/plots",
            "/retrain (POST)",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None}


def _ensure_expected_columns(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    try:
        preproc = model.named_steps.get("preprocesador")
    except Exception:
        preproc = None
    if preproc is None or not hasattr(preproc, "transformers_"):
        return df
    cols_expected: List[str] = []
    cat_cols: List[str] = []
    for name, _, cols in preproc.transformers_:
        try:
            if isinstance(cols, (list, tuple)):
                cols_expected.extend(list(cols))
                if name.lower().startswith("categor"):
                    cat_cols.extend(list(cols))
        except Exception:
            continue
    for c in cols_expected:
        if c not in df.columns:
            df[c] = "Unknown" if c in cat_cols else 0
    return df


# Base data directory (project-root relative). Useful when the server
# is started from a different working directory (e.g., inside Docker).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _select_data_dir() -> Path:
    # Try several candidate locations for outputs/data to be tolerant
    candidates = [
        PROJECT_ROOT / "outputs" / "data",
        Path("outputs") / "data",
        PROJECT_ROOT.parent / "outputs" / "data",
    ]
    for c in candidates:
        try:
            if c.exists():
                logger.info(f"Using data dir: {c}")
                return c
        except Exception:
            continue
    # Fallback to the project-root location even if it doesn't exist yet
    fallback = PROJECT_ROOT / "outputs" / "data"
    logger.warning(f"No existing data dir found; defaulting to {fallback}")
    return fallback

DATA_DIR = _select_data_dir()


def require_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


@app.post("/predict")
def predict(request: PredictionRequest, _=Depends(require_api_key)):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    if MODEL is None:
        logger.error("Predict called but model not loaded")
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    if not request.records:
        raise HTTPException(status_code=400, detail="`records` vacío")
    try:
        df = pd.DataFrame(request.records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if "Date" not in df.columns:
        df["Date"] = pd.Timestamp.now().normalize()
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Units Sold" not in df.columns:
        df["Units Sold"] = 0
    df = agregar_caracteristicas(df)
    df = _ensure_expected_columns(df, MODEL)
    try:
        with PREDICT_TIME.time():
            preds = MODEL.predict(df)
    except Exception as exc:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info(f"Predicted {len(preds)} rows")
    return {"predictions": preds.tolist(), "n": len(preds)}


@app.get("/model/version")
def model_version() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="metadata not found")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/reports/detail")
def report_detail(rows: int = 100) -> Dict[str, Any]:
    ruta = DATA_DIR / "pronostico_detalle.csv"
    logger.debug(f"Checking report detail at {ruta}")
    if not ruta.exists():
        logger.warning(f"Report detail not found at {ruta}")
        raise HTTPException(status_code=404, detail="not found")
    df = pd.read_csv(ruta)
    return {"rows": min(rows, len(df)), "data": df.head(rows).to_dict(orient="records")}


@app.get("/reports/season")
def report_season() -> Dict[str, Any]:
    ruta = DATA_DIR / "pronostico_temporada.csv"
    logger.debug(f"Checking report season at {ruta}")
    if not ruta.exists():
        logger.warning(f"Report season not found at {ruta}")
        raise HTTPException(status_code=404, detail="not found")
    df = pd.read_csv(ruta)
    return {"rows": len(df), "data": df.to_dict(orient="records")}


@app.get("/plots")
def list_plots() -> Dict[str, Any]:
    base = PROJECT_ROOT / "outputs" / "plots"
    if not base.exists():
        logger.debug(f"Plots directory not found at {base}")
        return {"plots": []}
    items = [p.as_posix() for p in base.rglob("*.png")]
    return {"plots": items}


def _run_retrain_subprocess():
    try:
        logger.info("Lanzando proceso de reentrenamiento")
        # Ejecuta script de entrenamiento en un proceso separado
        subprocess.Popen([sys.executable, "-m", "src.train.save_model"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        logger.exception("retrain subprocess failed")


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks, _=Depends(require_api_key)):
    background_tasks.add_task(_run_retrain_subprocess)
    return {"status": "started"}


@app.get("/metrics")
def metrics() -> Response:
    # Exponer métricas Prometheus
    if not METRICS_ENABLED:
        raise HTTPException(status_code=501, detail="metrics not enabled")
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
