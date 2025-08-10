# python
import os
import pandas as pd
from src.logger import logger
from Config import dbConfig as config
import io

def read_kaggle_dataset():
    # Defer Kaggle import to avoid auth during module import
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "Kaggle API is required to read the dataset but could not be imported."
        ) from e

    dataset = 'arshid/iris-flower-dataset'
    path = 'data'
    os.makedirs(path, exist_ok=True)

    # Set env vars only if provided; otherwise Kaggle will fall back to its default paths
    with open('./root/.config/kaggle/kaggle.json', 'r') as file:
        data = json.load(file)
        if 'kaggle_username' in data:
            os.environ['KAGGLE_USERNAME'] = data['kaggle_username']
        if 'kaggle_key' in data:
            os.environ['KAGGLE_KEY'] = data['kaggle_key']

    #if getattr(config, "kaggle_username", None):
    #    os.environ['KAGGLE_USERNAME'] = config.kaggle_username
    #if getattr(config, "kaggle_key", None):
    #    os.environ['KAGGLE_KEY'] = config.kaggle_key

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
