from store_sales.exception import CustomException
from store_sales.logger import logging
import os, sys
from store_sales.utils.utils import read_yaml_file
import pandas as pd
import collections


class IngestedDataValidation:

    def __init__(self, validate_path, schema_path):
        try:
            logging.info("*****************Ingested data validation Started****************")
            self.validate_path = validate_path
            self.schema_path = schema_path
            self.data = read_yaml_file(self.schema_path)
        except Exception as e:
            raise CustomException(e,sys) from e

    def validate_filename(self, file_name)->bool:
        try:
            logging.info(f"Checking filename")
            print(self.data["FileName"])
            schema_file_name = self.data['FileName']
            if schema_file_name == file_name:
                return True
            else:
                return False
            logging.info(f"filename Checked successfully")
        except Exception as e:
            raise CustomException(e,sys) from e

    def validate_column_length(self)->bool:
        try:
            logging.info(f"Checking validate_columns_length")
            df = pd.read_csv(self.validate_path)
            if(df.shape[1] == self.data['NumberOfColumns']):
                return True
            else:
                return False
            logging.info(f"validate_columns_length successfully")

        except Exception as e:
            raise CustomException(e,sys) from e

    def missing_values_whole_column(self)->bool:
        try:
            logging.info(f"Checking missing_values_whole_columns")
            df = pd.read_csv(self.validate_path)
            count = 0
            for columns in df:
                if (len(df[columns]) - df[columns].count()) == len(df[columns]):
                    count+=1
            return True if (count == 0) else False

            logging.info(f"missing_values_whole_columns checked successfully")
        except Exception as e:
            raise CustomException(e,sys) from e

    def replace_null_values_with_null(self)->bool:
        try:
            logging.info(f"Checking replacinig_null_values_with_null")
            df = pd.read_csv(self.validate_path)
            df.fillna('NULL',inplace=True)
            logging.info(f"replacinig_null_values_with_null is sucessfully")
        except Exception as e:
            raise CustomException(e,sys) from e

    
    def check_column_names(self)->bool:
        try:
            logging.info(f"Checking check_column_names") 
            df = pd.read_csv(self.validate_path)
            df_column_names = df.columns
            schema_column_names = list(self.data['ColumnNames'].keys())

            return True if (collections.Counter(df_column_names) == collections.Counter(schema_column_names)) else False
            logging.info(f"Checked check_column_names sucessfully") 
        except Exception as e:
            raise CustomException(e,sys) from e