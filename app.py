# Python
import os
import sys
import threading
import time
from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

from pydantic import BaseModel, ValidationError, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.logger import logger
from src.exception import CustomException
from src.iris_classification import data_ingestion
from src import train

app = Flask(__name__)

# --------------------
# Pydantic validation
# --------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "sepal_length": self.sepal_length,
                "sepal_width": self.sepal_width,
                "petal_length": self.petal_length,
                "petal_width": self.petal_width,
            }
        ])

# --------------------
# Prometheus metrics
# --------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["endpoint", "method"],
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["outcome"],  # species name or "error"
)

class DummyModel:
    """Simple fallback for CI/tests when MLflow model isn't available."""
    def predict(self, X):
        # Return a constant, length matches input rows
        return np.array(["setosa"] * len(X))

@lru_cache(maxsize=1)
def get_model():
    """
    Lazily load model.
    - If DISABLE_MLFLOW_MODEL=1 (CI/tests), use DummyModel.
    - Otherwise, try MLflow. On failure, fall back to DummyModel so app stays responsive.
    """
    if os.getenv("DISABLE_MLFLOW_MODEL") == "1" or os.getenv("FLASK_ENV") == "testing":
        logger.info("Using DummyModel (MLflow model loading disabled).")
        return DummyModel()

    try:
        # Optional: allow overriding both tracking and model URI from env
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/iris_best_model@production")
        logger.info(f"Loading MLflow model from URI: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.warning(f"Failed to load MLflow model, falling back to DummyModel: {e}")
        return DummyModel()

@app.before_request
def _before_request():
    # Start timer for latency measurement
    request._start_time = time.perf_counter()

@app.after_request
def _after_request(response):
    try:
        endpoint = request.endpoint or "unknown"
        method = request.method
        status = str(response.status_code)
        REQUEST_COUNT.labels(
            endpoint=request.path,
            method=request.method,
            status=response.status_code
        ).inc()
        if hasattr(request, "_start_time"):
            elapsed = time.perf_counter() - request._start_time
            REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)
    except Exception:
        # Metrics should never break the app
        pass
    return response

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/')
def home():
    grafana_cloud_url = os.getenv('GRAFANA_CLOUD_URL')
    return render_template('home.html', grafana_cloud_url=grafana_cloud_url)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = data.get('data')
        if input_data is None:
            raise ValueError("Missing 'data' in request body")
        try:
            validated = IrisInput(**input_data)
        except ValidationError as ve:
            PREDICTION_COUNT.labels(outcome="error").inc()
            return jsonify({'error': ve.errors()} )

        input_df = validated.to_df()
        prediction = get_model().predict(input_df)
        species = prediction.tolist()[0]
        PREDICTION_COUNT.labels(outcome=str(species)).inc()
        return jsonify({'prediction': species})
    except Exception as e:
        PREDICTION_COUNT.labels(outcome="error").inc()
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept both JSON and form submissions
        payload = request.get_json(silent=True) or {}

        validated = None
        if payload:
            # Support both {"data": {...}} and direct feature dict
            input_obj = payload.get("data", payload)
            try:
                validated = IrisInput(**{k: float(v) for k, v in input_obj.items()})
            except (ValueError, ValidationError) as ve:
                PREDICTION_COUNT.labels(outcome="error").inc()
                return render_template("home.html", prediction_text=f"Error: {ve}")
        else:
            # Fallback to form fields named exactly as features
            form_dict = {k: v for k, v in request.form.items()}
            try:
                validated = IrisInput(**{k: float(v) for k, v in form_dict.items()})
            except (ValueError, ValidationError) as ve:
                PREDICTION_COUNT.labels(outcome="error").inc()
                return render_template(
                    "home.html",
                    prediction_text=(
                        f"Error: {ve}. Expected fields: sepal_length, sepal_width, petal_length, petal_width"
                    ),
                )

        input_df = validated.to_df()
        prediction = get_model().predict(input_df)[0]
        PREDICTION_COUNT.labels(outcome=str(prediction)).inc()
        return render_template("home.html", prediction_text=f"The IRIS species is {prediction}")
    except Exception as e:
        PREDICTION_COUNT.labels(outcome="error").inc()
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

# --------------------
# New data ingestion and retraining endpoints
# --------------------

# Simple in-process training status
TRAIN_STATUS = {
    "running": False,
    "started_at": None,
    "completed_at": None,
    "last_accuracy": None,
    "last_error": None,
}
_TRAIN_LOCK = threading.Lock()


def _retrain_async():
    try:
        with _TRAIN_LOCK:
            TRAIN_STATUS.update({
                "running": True,
                "started_at": time.time(),
                "completed_at": None,
                "last_error": None,
            })
        logger.info("Starting async model training...")
        best_acc = train.main()
        # Refresh cached model so subsequent requests use the new one
        try:
            get_model.cache_clear()
        except Exception:
            pass
        with _TRAIN_LOCK:
            TRAIN_STATUS.update({
                "running": False,
                "completed_at": time.time(),
                "last_accuracy": float(best_acc) if best_acc is not None else TRAIN_STATUS.get("last_accuracy"),
            })
        logger.info("Async training finished.")
    except Exception as e:
        with _TRAIN_LOCK:
            TRAIN_STATUS.update({
                "running": False,
                "completed_at": time.time(),
                "last_error": str(e),
            })
        logger.error(f"Retraining failed: {e}")

@app.route('/ingest', methods=['POST'])
def ingest_new_data():
    """
    Accept new Iris records and append to Data/new_data.csv, then trigger retraining in background.
    Body can be one of:
    - {"data": {<IrisInput>}}
    - {"data": [{<IrisInput>}, {<IrisInput>}, ...]}
    """
    try:
        payload = request.get_json(force=True) or {}
        data = payload.get("data")
        if data is None:
            raise ValueError("Missing 'data' in request body")

        items = data if isinstance(data, list) else [data]
        records = []
        for item in items:
            try:
                rec = IrisInput(**item)
                records.append(rec)
            except ValidationError as ve:
                return jsonify({"error": ve.errors()})

        df = pd.DataFrame([{
            "sepal_length": r.sepal_length,
            "sepal_width": r.sepal_width,
            "petal_length": r.petal_length,
            "petal_width": r.petal_width,
        } for r in records])

        os.makedirs("Data", exist_ok=True)
        target_path = os.path.join("Data", "new_data.csv")
        header = not os.path.exists(target_path)
        df.to_csv(target_path, mode='a', header=header, index=False)

        # Trigger background retraining
        threading.Thread(target=_retrain_async, daemon=True).start()
        return jsonify({"status": "accepted", "ingested_rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    try:
        with _TRAIN_LOCK:
            if TRAIN_STATUS.get("running"):
                return jsonify({
                    "status": "already_running",
                    "status_url": "/retrain/status",
                    "detail": "Training is already in progress"
                })
        threading.Thread(target=_retrain_async, daemon=True).start()
        return jsonify({
            "status": "accepted",
            "status_url": "/retrain/status"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/retrain/status', methods=['GET'])
def retrain_status():
    # Lightweight training status for the UI to poll
    try:
        with _TRAIN_LOCK:
            return jsonify(TRAIN_STATUS.copy())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/model/metrics', methods=['GET'])
def model_metrics():
    """
    Return the latest persisted training metrics and confusion matrix
    from Data/last_metrics.json. This is updated after each training run.
    """
    try:
        metrics_path = os.path.join('Data', 'last_metrics.json')
        if not os.path.exists(metrics_path):
            return jsonify({"message": "Metrics not available yet."})
        import json
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/logs', methods=['GET'])
def view_logs():
    """
    Render the latest log file from the logs directory.
    """
    try:
        logs_dir = 'logs'
        if not os.path.isdir(logs_dir):
            return Response('<h3>No logs directory found.</h3>', mimetype='text/html')
        # Find latest file by mtime
        candidates = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]
        if not candidates:
            return Response('<h3>No log files found.</h3>', mimetype='text/html')
        latest = max(candidates, key=os.path.getmtime)
        # Read last N bytes to avoid huge files
        try:
            with open(latest, 'rb') as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                # Read up to last 200KB
                read_size = min(size, 200 * 1024)
                fh.seek(-read_size, os.SEEK_END)
                content = fh.read().decode('utf-8', errors='replace')
        except Exception:
            with open(latest, 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read()
        html = f"""
        <html><head><title>Logs - Latest</title>
        <meta charset='utf-8'>
        <style> body{{font-family:system-ui,Segoe UI,Arial; padding:16px; background:#0b0f14; color:#e9ecef;}} pre{{background:#0e141b;color:#e9ecef;padding:12px;border-radius:6px;white-space:pre-wrap;}} a{{text-decoration:none;color:#9ccaff}} h2{{color:#e9ecef}} </style>
        </head>
        <body>
        <h2>Latest log: {os.path.basename(latest)}</h2>
        <div><a href='/'>← Back to Home</a></div>
        <pre>{content}</pre>
        </body></html>
        """
        return Response(html, mimetype='text/html')
    except Exception as e:
        return Response(f"<h3>Error reading logs: {e}</h3>", mimetype='text/html')

@app.route('/logs/tail', methods=['GET'])
def tail_logs():
    """Return the tail of the latest log file as JSON for live display in the UI."""
    try:
        logs_dir = 'logs'
        if not os.path.isdir(logs_dir):
            return jsonify({"message": "No logs directory found.", "content": ""})
        candidates = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]
        if not candidates:
            return jsonify({"message": "No log files found.", "content": ""})
        latest = max(candidates, key=os.path.getmtime)
        try:
            with open(latest, 'rb') as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                read_size = min(size, 200 * 1024)
                fh.seek(-read_size, os.SEEK_END)
                content = fh.read().decode('utf-8', errors='replace')
        except Exception:
            with open(latest, 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read()
        return jsonify({
            "filename": os.path.basename(latest),
            "updated_at": os.path.getmtime(latest),
            "size": os.path.getsize(latest),
            "content": content
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    logger.info("Execution started")
    try:
        logger.info("Starting data ingestion process...")
        data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")

        logger.info("Starting model training process...")
        train.main()
        logger.info("Model training completed.")
        app.run(debug=True, port=8000)
    except Exception as e:
        logger.error("An exception occurred during pipeline execution")
        raise CustomException(e, sys)