import os
import sys
from six.moves import urllib
import numpy as np
import shutil
import pandas as pd
from store_sales.logger import logging
from store_sales.exception import CustomException
from store_sales.constant import *

from store_sales.utils.utils import read_yaml_file
from store_sales.data_access.goog_final import GoogleDriveDownloader
from store_sales.entity.config_entity import DataIngestionConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # Disable warning for setting with copy
# Set low_memory option to False
pd.options.mode.use_inf_as_na = False


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.config_info  = read_yaml_file(file_path=CONFIG_FILE_PATH)
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomException(e, sys) from e
    '''
    def get_data_from_mongo_DB(self) -> str:
        try:
            
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)

            file_name = FILE_NAME

            raw_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(
                f"Downloading file from Mongo DB into :[{raw_file_path}]")
            
            data=mongodata()
            # Storing mongo data ccs file to raw directoy 
            data.export_collection_as_csv(collection_name=COLLECTION_NAME,database_name=DATABASE_NAME,file_path=raw_file_path)
            
            logging.info(
                f"File :[{raw_file_path}] has been downloaded successfully.")
            return raw_file_path

        except Exception as e:
            raise CustomException(e, sys) from e
        
        
        
        '''
    def get_csv_from_google_drive(self,file_url:list,file_name:list):
        raw_data_dir = self.data_ingestion_config.raw_data_dir
        
        # Make Raw data Directory
        os.makedirs(raw_data_dir, exist_ok=True)
        destination_folder = raw_data_dir
        os.makedirs(destination_folder, exist_ok=True)

        downloader = GoogleDriveDownloader(file_url, file_name, destination_folder)
        downloader.download()
            
        return os.path.join(destination_folder,file_name)
        
    def initiate_data_ingestion(self):
        try:
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            file_path=data_ingestion_info[FILE_PATH]
            file_name=data_ingestion_info[FILE_NAME]
            
            raw_data_dir=self.data_ingestion_config.raw_data_dir
            
            # Create the raw_data_dir directory if it doesn't exist
            os.makedirs(raw_data_dir, exist_ok=True)

            # Define the destination path
            destination_path = os.path.join(raw_data_dir, file_name)

            # Copy the file to the raw_data_dir and rename it
            shutil.copyfile(file_path, destination_path)

            # Print the success message
            print("File copied and renamed successfully!")
            print("Destination file path:", destination_path)
            message="Data Ingestion complete"
            
            logging.info("Data Ingestion Artifact")
            logging.info(f"File Path : {file_path}")
            logging.info(f"File Name: {file_name}")
            logging.info(f"File Ingested File Path: {destination_path}")
            
            data_ingestion_artifact=DataIngestionArtifact(
                message=message,
                ingestion_file_path=destination_path
                )
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)from e