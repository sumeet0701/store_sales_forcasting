import os  
import sys 
from store_sales.config import *
from store_sales.entity.config_entity import DataIngestionConfig
from store_sales.entity.config_entity import DataValidationConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact
from store_sales.entity.artifact_entity import DataValidationArtifact
from store_sales.entity.raw_data_validation import IngestedDataValidation
from store_sales.config.Configuration import Configuration
from store_sales.exception import CustomException
from store_sales.logger import logging
from store_sales.utils.utils import read_yaml_file

import shutil
from store_sales.constant import *
import pandas as pd
import json

from sklearn.model_selection import train_test_split

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

pd.options.mode.chained_assignment = None  # Disable warning for setting with copy
# Set low_memory option to False
pd.options.mode.use_inf_as_na = False


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n") 
            
            # Creating_instance           
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
            # Schema_file_path
            self.schema_path = self.data_validation_config.schema_file_path
            
            # creating instance for row_data_validation
            self.train_data = IngestedDataValidation(
                                validate_path=self.data_ingestion_artifact.ingestion_file_path, schema_path=self.schema_path)
            
            # Data_ingestion_artifact--->Unvalidated train and test file path
            self.ingested_file_path = self.data_ingestion_artifact.ingestion_file_path

            # Data_validation_config --> file paths to save validated_data
            self.validated_file_path = self.data_validation_config.file_path
            
        
        except Exception as e:
            raise CustomException(e,sys) from e



    def isFolderPathAvailable(self) -> bool:
        try:

             # True means avaliable false means not avaliable
             
            isfolder_available = False
            datafile_path=self.ingested_file_path

            if os.path.exists(datafile_path):
                    isfolder_available = True
            return isfolder_available
        except Exception as e:
            raise CustomException(e, sys) from e     
      


        
    def is_Validation_successfull(self):
        try:
            validation_status = True
            logging.info("Validation Process Started")
            if self.isFolderPathAvailable() == True:
                train_filename = os.path.basename(
                    self.data_ingestion_artifact.ingestion_file_path)

                is_train_filename_validated = self.train_data.validate_filename(
                    file_name=train_filename)

                is_train_column_name_same = self.train_data.check_column_names()

                is_train_missing_values_whole_column = self.train_data.missing_values_whole_column()

                self.train_data.replace_null_values_with_null()
                 

                logging.info(
                    f"Train_set status|is Train filename validated?: {is_train_filename_validated}|is train column name validated?: {is_train_column_name_same}|whole missing columns?{is_train_missing_values_whole_column}")

                if is_train_filename_validated  & is_train_column_name_same & is_train_missing_values_whole_column:
                    ## Exporting Train.csv file 
                    # Create the directory if it doesn't exist
                    os.makedirs(self.validated_file_path, exist_ok=True)

                    # Copy the CSV file to the validated train path
                    shutil.copy(self.ingested_file_path, self.validated_file_path)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated  dataset to file: [{self.validated_file_path}]")
                                     
                    return validation_status,self.validated_file_path
                else:
                    validation_status = False
                    logging.info("Check yout Data! Validation Failed")
                    raise ValueError(
                        "Check your data! Validation failed")
                

            return validation_status,"NONE"
        except Exception as e:
            raise CustomException(e, sys) from e      

    def get_train_test_df(self):
        try:
            df = pd.read_csv(self.data_ingestion_artifact.ingestion_file_path)
            train_df , test_df = train_test_split(df, test_size= 0.3, random_state= 0)
            return train_df, test_df
        except Exception as e:
            raise CustomException(e,sys) from e
    def get_and_save_data_drift_report(self):
        try:
            logging.info("Generating data drift report.json file")
            profile = Profile(sections = [DataDriftProfileSection()])
            train_df, test_df = self.get_train_test_df()
            profile.calculate(train_df, test_df)
            
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent = 6)
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise CustomException(e,sys) from e   

    def save_data_drift_report_page(self):
        try:
            logging.info("Generating data drift report.html page")
            dashboard = Dashboard(tabs = [DataDriftTab()])
            train_df, test_df = self.get_train_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
            logging.info("Report.html page generation successful!!")
        except Exception as e:
            raise CustomException(e,sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            logging.info("Checking for Data Drift")
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise CustomException(e,sys) from e


    def initiate_data_validation(self):
        try:
            is_validated, validated_file_path = self.is_Validation_successfull()
            self.is_data_drift_found()
            if is_validated is True:
                message='Validated'
            else:
                message='Not Validated'


            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.schema_path,
                message=message,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                validated_file_path = validated_file_path  
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    def __del__(self):
        logging.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30}")