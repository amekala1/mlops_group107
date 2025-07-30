from src.logger import logger
from src.exception import CustomException
from src.iris_classification import data_ingestion
from src import train

import sys

if __name__ == "__main__":
    logger.info("Execution started")

    try:
        logger.info("Starting data ingestion process...")
        data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")

        logger.info("Starting model training process...")
        train.main()
        logger.info("Model training completed.")

    except Exception as e:
        logger.error("An exception occurred during pipeline execution")
        raise CustomException(e, sys)
