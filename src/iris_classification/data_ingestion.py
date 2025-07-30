import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from src.logger import logger
from Config import dbConfig as config
import io

def read_kaggle_dataset():
    dataset = 'arshid/iris-flower-dataset'
    path = 'data'
    os.makedirs(path, exist_ok=True)

    os.environ['KAGGLE_USERNAME'] = config.kaggle_username
    os.environ['KAGGLE_KEY'] = config.kaggle_key

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path, unzip=True)
    logger.info("Dataset downloaded from Kaggle.")

def export_as_dataframe():
    input_path = os.path.join('data', 'IRIS.csv')
    output_path = os.path.join('data', 'cleaned.csv')

    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info(buffer.getvalue())

    # Handle missing values
    df.dropna(inplace=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")

def initiate_data_ingestion():
    logger.info("Starting data ingestion process.")
    read_kaggle_dataset()
    export_as_dataframe()
    logger.info("Data ingestion completed.")
