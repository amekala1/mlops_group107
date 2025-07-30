import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.logger import logger
from src.exception import CustomException
from src.mlflow_utils import get_best_model

def load_data():
    try:
        raw_path = os.path.join("Data", "raw.csv")
        df = pd.read_csv(raw_path)
        return df
    except Exception as e:
        raise CustomException(e, sys)

def preprocess(df):
    try:
        X = df.drop(columns=["species"])
        y = df["species"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        raise CustomException(e, sys)

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    try:
        with mlflow.start_run():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            logger.info(f"{model_name} Accuracy: {acc}")
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", acc)

            # Infer input/output signature
            input_example = X_test.iloc[:1]
            signature = infer_signature(X_test, preds)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

            return acc, mlflow.active_run().info.run_id

    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess(df)

        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or use default
        mlflow.set_experiment("Iris_Classification")

        results = {}

        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=200)
        acc1, run_id1 = train_and_log_model(lr_model, "LogisticRegression", X_train, X_test, y_train, y_test)
        results[acc1] = run_id1

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100)
        acc2, run_id2 = train_and_log_model(rf_model, "RandomForest", X_train, X_test, y_train, y_test)
        results[acc2] = run_id2

        # Choose best and register
        best_acc = max(results.keys())
        best_run_id = results[best_acc]

        logger.info(f"Best Model Accuracy: {best_acc}, Run ID: {best_run_id}")
        get_best_model(run_id=best_run_id, model_name="iris_best_model")

    except Exception as e:
        logger.error("Error in training pipeline")
        raise CustomException(e, sys)
