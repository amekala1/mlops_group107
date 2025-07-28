import os
import pandas as pd
#from exception import CustomException
from src.logger import logging
from io import StringIO 

#Connect to the PostgreSQL database
#import psycopg2
import csv

#Read the dataset from Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import sys

from Config import dbConfig as config

# Set environment variables for Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = config.kaggle_username
os.environ['KAGGLE_KEY'] = config.kaggle_key      


#Read the dataset from Kaggle
def readKaggleDataset():

    dataset = 'arshid/iris-flower-dataset'
    path = 'Data'

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path, unzip=True)

def exportAsDataFrame():

    #Load date into Pandas dataframe and explore it 
    test_data_path:str=os.path.join('Data','IRIS.csv')
    raw_data_path:str=os.path.join('Data','raw.csv')
    df_copy = pd.read_csv(test_data_path)
    logging.info(len(df_copy)) 

    df = df_copy.copy()

    logging.info("row x columns of data   ", df.shape) # row x columns of data
    logging.info("size of data   ", df.size) # size of data

    logging.info("##################################################################")
    df.info()
    logging.info("##################################################################")

    # Do some data preparation
    #Find out missing values
    miss = df.isnull().sum().sort_values(ascending = False).head()
    #Drop missing values
    df.dropna(inplace=True)
    #logging.info("Number of missing values : AFTER  ", sum(df.isnull().sum()>0))

    #Move the prepared dataset to a new file 
    df.to_csv(raw_data_path, header=True, index=False, encoding='utf-8')
        
    logging.info("Raw Data file exported!")
        

def initiate_data_ingestion():
    logging.info("Entering data ingestion method")
    #try:
    readKaggleDataset()
    exportAsDataFrame()
     
    logging.info("Data ingestion completed successfully")
        