# Python
import os
import sys
from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.logger import logger
from src.exception import CustomException
from src.iris_classification import data_ingestion
from src import train

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = data['data']  # {"sepal_length": 5.1, "sepal_width": 3.5, ...}
        input_df = pd.DataFrame([input_data])
        prediction = get_model().predict(input_df)
        return jsonify({'prediction': prediction.tolist()[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = [float(x) for x in request.form.values()]
        input_df = pd.DataFrame([form_values], columns=[
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ])
        prediction = get_model().predict(input_df)[0]
        return render_template("home.html", prediction_text=f"The IRIS species is {prediction}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

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