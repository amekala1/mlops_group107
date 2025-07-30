import mlflow
import os
from src.logger import logger
import mlflow
from mlflow.tracking import MlflowClient
from src.logger import logger
from src.exception import CustomException
import sys

def set_mlflow_tracking(uri: str = "mlruns"):
    """
    Set the MLflow tracking URI.
    By default, uses local directory. For remote tracking server, pass URI.
    """
    try:
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to: {uri}")
    except Exception as e:
        logger.error(f"Failed to set MLflow tracking URI: {e}")
        raise e

def start_experiment(experiment_name: str):
    """
    Set or create MLflow experiment.
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise e

def register_model(model_uri: str, model_name: str):
    """
    Register model to the MLflow model registry.
    Example: register_model("runs:/<run_id>/model", "best_model")
    """
    try:
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered: {result.name}, Version: {result.version}")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise e


def get_best_model(run_id: str, model_name: str):
    """
    Registers the model from the given run_id as the best model in the MLflow Model Registry.
    """
    try:
        logger.info(f"Registering model: run_id={run_id}, name={model_name}")

        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(f"Model registered: name={model_name}, version={result.version}")
        return result

    except Exception as e:
        logger.error("Failed to register best model")
        raise CustomException(e, sys)
