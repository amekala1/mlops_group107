# Iris MLOps App

How to run:

1. Install dependencies:
   - pip install -r requirements.txt
2. Run the app:
   - python app.py
3. In another terminal, run MLflow UI to inspect runs:
   - mlflow ui
4. Download Prometheus windows application in order to get it connect to grafana cloud. 
   - Run the prometheus server in another terminal: "EX: (path to prometheus.exe)prometheus.exe --config.file="monitoring\\prometheus.yml
6. Create a graphana instance and update the dashboard with the url of the prometheus server.
7. Run the app:
   - python app.py
8. Open localhost:8000 in your browser. for all the endpoints and UI to interact with the app.


Environment:
- Optionally set MLFLOW_TRACKING_URI and MLFLOW_MODEL_URI.
- To disable MLflow model loading for tests, set DISABLE_MLFLOW_MODEL=1.

Input validation:
- All inputs are validated using Pydantic. Features required:
  sepal_length, sepal_width, petal_length, petal_width (floats, >= 0)
- Endpoints:
  - POST /predict_api with JSON: {"data": {"sepal_length": 5.1, ...}}
  - POST /predict (form)

Prometheus metrics:
- Metrics exposed at /metrics (Prometheus text format).
- Provided sample configs:
  - monitoring/prometheus.yml (scrapes http://127.0.0.1:8000/metrics)
  - monitoring/grafana-dashboard.json (import into Grafana)

New endpoints:
- GET /model/metrics: Latest training metrics (accuracy, precision_macro, recall_macro, f1_macro) and confusion matrix data.

Using Prometheus on Windows (native app):
1) Download the latest Prometheus for Windows (prometheus-x.y.z.windows-amd64.zip) from https://prometheus.io/download/
2) Unzip it, e.g., to C:\Tools\prometheus
3) Copy monitoring\prometheus.yml from this repo into the Prometheus folder and overwrite its prometheus.yml, or start Prometheus with:
   prometheus.exe --config.file="C:\\path\\to\\repo\\monitoring\\prometheus.yml"
4) Start Prometheus by double-clicking prometheus.exe or running from PowerShell. Open http://127.0.0.1:9090 in your browser.
5) Verify on the Status -> Targets page that iris-flask-app is UP and scraping http://127.0.0.1:8000/metrics.
6) In the 'Graph' tab, try queries like:
   - rate(http_requests_total[1m])
   - histogram_quantile(0.95, sum(rate(http_request_latency_seconds_bucket[5m])) by (le, endpoint))
   - sum(rate(predictions_total[5m])) by (outcome)


Model re-training on new data:
- POST /ingest with JSON body:
  {"data": {<Iris features>}} or {"data": [{...}, {...}]}
  Appends to Data/new_data.csv and triggers background retraining.
- POST /retrain triggers background retraining without ingesting.
- After training completes, the model cache is refreshed automatically.
- UI experience: When you click "POST /retrain" in the Retrain tab, a spinner shows while training runs. The page polls /retrain/status; when complete, you’ll be redirected to the Predict tab where the last training accuracy and completion time are displayed.

Sample API calls:
- Predict via JSON (PowerShell)
  $body = @{ data = @{ sepal_length=5.1; sepal_width=3.5; petal_length=1.4; petal_width=0.2 } } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict_api" -Method Post -Body $body -ContentType "application/json"

- Ingest new data (PowerShell)
  $ing = @{ data = @(@{ sepal_length=6.1; sepal_width=2.8; petal_length=4.7; petal_width=1.2 }) } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest" -Method Post -Body $ing -ContentType "application/json"

- Trigger retrain and check status (PowerShell)
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/retrain" -Method Post
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/retrain/status" -Method Get

- Get latest model metrics (JSON)
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/metrics" -Method Get

- View metrics (Prometheus scrape output)
  Invoke-WebRequest -Uri "http://127.0.0.1:8000/metrics" | Select-Object -ExpandProperty Content

- curl examples
  curl -X POST http://127.0.0.1:8000/predict_api ^
    -H "Content-Type: application/json" ^
    -d "{\"data\": {\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}}"
  curl -X POST http://127.0.0.1:8000/ingest ^
    -H "Content-Type: application/json" ^
    -d "{\"data\":[{\"sepal_length\":6.1,\"sepal_width\":2.8,\"petal_length\":4.7,\"petal_width\":1.2}]}"
  curl -X POST http://127.0.0.1:8000/retrain
  curl http://127.0.0.1:8000/retrain/status
  curl http://127.0.0.1:8000/model/metrics
  curl http://127.0.0.1:8000/metrics

Notes:
- When running in PyCharm, add root/creds.env to the run configuration (do not commit this file).

Running from PyCharm (Windows):
- Open the project in PyCharm.
- File -> Settings -> Project -> Python Interpreter: select/create a venv with Python 3.10+ and install requirements.txt.
- Run Configuration:
  - Script: app.py
  - Parameters: (leave empty)
  - Working directory: project root
  - Environment variables (optional):
    - DISABLE_MLFLOW_MODEL=1 (fast dev without MLflow model loading)
    - MLFLOW_TRACKING_URI=http://127.0.0.1:5000 (if using MLflow UI)
  - Before launch: you can create a second Run config for `mlflow ui` and start it separately.
- Click Run. App listens on http://127.0.0.1:8000

Run with Docker (Container):
- Build image (note: DockerFile filename is used):
  - docker build -t iris-mlops -f DockerFile .
- Run container mapping port 8000:
  - docker run --rm -p 8000:8000 -e DISABLE_MLFLOW_MODEL=1 --name iris iris-mlops
- Optional: If you run MLflow UI on host (port 5000), the app logs to it by default; or pass MLFLOW_TRACKING_URI env var.
- Prometheus (Docker Desktop): use target host.docker.internal:8000 in your Prometheus config inside Docker.

CI/CD pipeline (example with GitHub Actions):
- Create .github/workflows/ci.yml with:

  name: CI
  on:
    push:
      branches: [ main ]
    pull_request:
      branches: [ main ]
  jobs:
    test:
      runs-on: windows-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.12'
        - name: Install deps
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
        - name: Run tests
          env:
            DISABLE_MLFLOW_MODEL: '1'   # avoid mlflow model loading during CI
          run: |
            pytest -q || pytest -q -k "not kaggle"
    docker:
      needs: test
      runs-on: ubuntu-latest
      permissions:
        contents: read
        packages: write
      steps:
        - uses: actions/checkout@v4
        - name: Build Docker image
          run: docker build -t ghcr.io/${{ github.repository }}:latest -f DockerFile .
        # - name: Login and push (optional)
        #   uses: docker/login-action@v3
        #   with:
        #     registry: ghcr.io
        #     username: ${{ github.actor }}
        #     password: ${{ secrets.GITHUB_TOKEN }}
        # - name: Push
        #   run: docker push ghcr.io/${{ github.repository }}:latest

Flask API details:
- POST /predict_api
  - Body: {"data": {"sepal_length": float, "sepal_width": float, "petal_length": float, "petal_width": float}}
  - Response: {"prediction": "Iris-setosa"} or {"error": ...}
- POST /predict (form)
  - Fields: sepal_length, sepal_width, petal_length, petal_width
  - Response: HTML page with prediction text.
- POST /ingest
  - Body: {"data": {...}} or {"data": [{...}, {...}]}
  - Action: appends to Data/new_data.csv, triggers background retraining.
  - Response: {"status": "accepted", "ingested_rows": n} or {"error": ...}
- POST /retrain
  - Triggers background retraining.
  - Response: {"status": "accepted", "status_url": "/retrain/status"}
- GET /retrain/status
  - Returns { running, started_at, completed_at, last_accuracy, last_error }
- GET /model/metrics
  - Returns latest metrics JSON including accuracy, precision_macro, recall_macro, f1_macro, labels, confusion_matrix, updated_at
- GET /metrics
  - Prometheus text exposition format with http_requests_total, http_request_latency_seconds, predictions_total



## Grafana Cloud (remote_write)

If you use Grafana Cloud, you can push your local Prometheus metrics to Grafana Cloud using remote_write.

1) Edit monitoring\prometheus.yml to include your Grafana Cloud Prometheus remote_write endpoint and credentials. This repo already includes an example with:

   global:
     scrape_interval: 60s
   remote_write:
     - url: https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push
       basic_auth:
         username: 2610093
         password: <YOUR_API_TOKEN>
   scrape_configs:
     - job_name: 'iris-flask-app'
       metrics_path: /metrics
       static_configs:
         - targets: ['127.0.0.1:8000']

   Important: Replace <YOUR_API_TOKEN> with the Grafana Cloud access token you created for Prometheus remote write. Treat this token as a secret. Prefer using a separate untracked file or environment variables in production.

2) Start Prometheus with this config (Windows PowerShell):
   cd C:\\path\\to\\prometheus-folder
   .\\prometheus.exe --config.file="C:\\path\\to\\repo\\monitoring\\prometheus.yml"

3) Wait at least one scrape interval (60s by default). Data appears in Grafana Cloud under Explore → Choose your Grafana Cloud Prometheus datasource.

4) Run queries in Grafana Cloud (examples):
   - rate(http_requests_total[1m])
   - histogram_quantile(0.95, sum(rate(http_request_latency_seconds_bucket[5m])) by (le, endpoint))
   - sum(rate(predictions_total[5m])) by (outcome)

5) Import the provided dashboard (monitoring\\grafana-dashboard.json) into Grafana Cloud:
   - Dashboards → Import → Upload JSON file → select monitoring\\grafana-dashboard.json
   - When prompted, select your Grafana Cloud Prometheus datasource and Import.

Notes
- Ensure your Flask app is running locally at http://127.0.0.1:8000; Prometheus scrapes /metrics from there.
- If you also run a local Grafana (port 3000) with a local Prometheus (port 9090), that setup is independent from Grafana Cloud. You can keep both; the remote_write simply ships a copy of your metrics to Grafana Cloud.
- If you need a faster update cadence in Grafana Cloud, reduce scrape_interval (e.g., 15s) but be mindful of rate limits and cost.
