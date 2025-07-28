from src.logger import logging
from src.exception import CustomException
from src.iris_classification import data_ingestion

import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:

        data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

