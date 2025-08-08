import mlflow
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.iris_classification import data_ingestion
from src import train
from flask import Flask, render_template, request, jsonify, make_response
import pickle
import numpy as np

import sys
model = mlflow.pyfunc.load_model("models:/iris_best_model@production")

app=Flask(__name__)
## Load the model
#regmodel=pickle.load(open('Data/regmodel.pkl','rb'))
#scalar=pickle.load(open('Data/scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = data['data']  # {"sepal_length": 5.1, "sepal_width": 3.5, ...}
        print("Data received for prediction:", input_data)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)

        # Return JSON response
        return jsonify({'prediction': prediction.tolist()[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float list
        form_values = [float(x) for x in request.form.values()]
        input_df = pd.DataFrame([form_values], columns=[
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ])

        # Predict
        prediction = model.predict(input_df)[0]

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
        app.run(debug=True,port=8000)

    except Exception as e:
        logger.error("An exception occurred during pipeline execution")
        raise CustomException(e, sys)