from ast import expr_context
from operator import index
from store_sales.logger import logging
from store_sales.exception import CustomException
from store_sales.entity.config_entity import DataIngestionConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact
from store_sales.components.db_operation import MongoDB
import pandas as pd
import numpy as np
import os,sys
import zipfile
from six.moves import urllib
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"\n{'*'*20} Data Ingestion log started {'*'*20}\n")
            self.data_ingestion_config = data_ingestion_config

            # Creating connection with the DB
            self.db = MongoDB()

        except Exception as e:
            raise CustomException(e,sys) from e

    def download_data(self):
        """
        Downloads the zipped dataset from the given url and save it to the specified path.
        """
        try:
            # Extracting remote url to download dataset files
            download_url = self.data_ingestion_config.dataset_download_url

            # folder location to download zipped file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir

            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
            os.makedirs(tgz_download_dir,exist_ok=True)

            #file_name = os.path.basename(download_url)
            file_name = "store_sales.zip"
            tgz_file_path = os.path.join(tgz_download_dir,file_name)

            logging.info(f"Downloading file from: [{download_url}] into : [{tgz_file_path}]")
            urllib.request.urlretrieve(download_url,tgz_file_path)
            logging.info(f"File: [{tgz_file_path}] has been downloaded successfully")

            return tgz_file_path

        except Exception as e:
            raise CustomException(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            # Folder location to extract the downloaded zipped dataset files
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting zipped file : [{tgz_file_path}] into dir: [{raw_data_dir}]")
            # Extarcting the files from zipped file
            zip_ref = zipfile.ZipFile(tgz_file_path)
            zip_ref.extractall(raw_data_dir)
            zip_ref.close()

            logging.info("Extraction completed successfully")

        except Exception as e:
            raise CustomException(e,sys) from e

    def data_merge_and_split(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir  # Location for extracted data files
            
            file_name = os.listdir(raw_data_dir)[0]
            data_file_path = os.path.join(raw_data_dir)
            
            # Creating collection in mongoDb for dumping data
            #self.db.create_and_check_collection()
            
            # Reading each data files and dumping it into DB
            for file in os.listdir(data_file_path):
                data = pd.read_csv(os.path.join(data_file_path,file))
                data_dict = data.to_dict("records")
                logging.info(f"Inserting file: [{file}] into DB")
                #self.db.insertall(data_dict)

            # fetching the data set from DB
            logging.info(f"Fetching entire data from DB")
            #dataframe = self.db.fetch_df()
            #dataframe.drop(columns = "_id",inplace=True)
            logging.info(f"Entire data fetched successfully from DB!!!")

            # Splitting the dataset into train and test data based on date indexing
            logging.info("Splitting Dataset into train and test")
            ingested_data = data
            logging.info(f"{ingested_data.columns}")

            logging.info("Inserting new Training Data into DB")
            #self.db.create_and_check_collection(coll_name="Training")
            #self.db.insertall(ingested_data.to_dict("records"))

            # Setting paths for train and test data
            ingested_data_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir,file_name)
            #test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)

            if ingested_data is not None:
                os.makedirs(self.data_ingestion_config.ingested_data_dir,exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{ingested_data_file_path}]")
                ingested_data.to_csv(ingested_data_file_path,index=False)



            data_ingestion_artifact = DataIngestionArtifact(Ingestion_file_path= ingested_data_file_path,
                                                            is_ingested=True,
                                                            message="Data ingestion completed successfully")
            logging.info(f"Data Ingestion Artifact: [{data_ingestion_artifact}]")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys) from e
    
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            tgz_file_path = self.download_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.data_merge_and_split()
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def __del__(self):
        logging.info(f"\n{'*'*20} Data Ingestion log completed {'*'*20}\n")