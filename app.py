from src.logger import logger
from src.exception import CustomException
from src.iris_classification import data_ingestion
from src import train
from flask import Flask, render_template, request, jsonify, make_response
import pickle
import numpy as np

import sys

app=Flask(__name__)
## Load the model
#regmodel=pickle.load(open('Data/regmodel.pkl','rb'))
#scalar=pickle.load(open('Data/scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    #output=regmodel.predict(new_data)
    #print(output[0])
    print("Data received for prediction:", data)
    #return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    #data=[float(x) for x in request.form.values()]
    #final_input=scalar.transform(np.array(data).reshape(1,-1))
    print("IRIS Classification API is called")
    #output=regmodel.predict(final_input)[0]
    #return render_template("home.html",prediction_text="The IRIS species is {}".format(output))

if __name__ == "__main__":
    logger.info("Execution started")

    try:
        logger.info("Starting data ingestion process...")
        #data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")

        logger.info("Starting model training process...")
        #train.main()
        logger.info("Model training completed.")
        app.run(debug=True)

    except Exception as e:
        logger.error("An exception occurred during pipeline execution")
        raise CustomException(e, sys)
