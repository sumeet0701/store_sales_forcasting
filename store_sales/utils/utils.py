import yaml
from store_sales.exception import CustomException
import os,sys
import dill
import pandas as pd
import numpy as np
import pymongo
from store_sales.constant import *
from store_sales.logger import logging

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise CustomException(e,sys)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as dictionary.

    Params:
    ---------------
    file_path (str) : file path for the yaml file
    """
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(file_path:str,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e

'''   
def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path )

        error_message = ""

        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe

    except Exception as e:
        raise CustomException(e, sys) from e
'''

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e

def save_data(file_path:str, data:pd.DataFrame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(file_path,index = None)
    except Exception as e:
        raise CustomException(e,sys) from e
    
def save_array_to_directory(array: np.array, 
                            directory_path: str, 
                            file_name: str, 
                            extension: str = '.npy'):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Add the extension to the file name
        file_name_with_extension = file_name + extension

        # Generate the file path
        file_path = os.path.join(directory_path, file_name_with_extension)

        # Save the array to the file path
        np.save(file_path, array)
    except Exception as e:
        CustomException(e,sys)

def save_image(image_path, image):
    """
    Save an image to a specified file path.
    
    Args:
        image_path (str): The file path to save the image.
        image (PIL.Image.Image): The image object to save.
    """
    try:
        image.save(image_path)
        print(f"Image saved successfully at {image_path}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")
