from store_sales.exception import CustomException
from store_sales.logger import logging
from store_sales.constant import *
from store_sales.config.Configuration import Configuration
from store_sales.utils.utils import read_yaml_file
from store_sales.utils.utils import load_object
from store_sales.utils.utils import save_object
from store_sales.entity.artifact_entity import ModelTrainerArtifact
from store_sales.entity.artifact_entity import DataTransformationArtifact

from sklearn.pipeline import Pipeline
from tkinter import E

import os
import sys
import shutil
import pandas as pd
import numpy as np
import joblib


BATCH_PREDICTION = "batch_prediction"
INSTANCE_PREDICTION="Instance_prediction"
input_file_path="outlier_clean.csv"
feature_engineering_file_path ="prediction_files/feat_eng.pkl"
transformer_file_path ="prediction_files/preprocessed.pkl"
model_file_path ="saved_model/model.pkl"


# Load the Preprocessor and Time series Model 
preprocesser = joblib.load("prediction_files/preprocessed.pkl")
model = joblib.load("saved_model/model.pkl")

# define mapping

STORE_TYPE  = {'A':0, "B":1, "C":2, "D":3, "E":4}
HOLIDAY_TYPE ={'Holiday':3, 'Event':2, 'Additional':0, 'Transfer':4, 'Work Day': 5, "Bridge":1}

class InstancePrediction:
    def __init__(self, store_type, store_nbr, holiday_type, onpromotion, oil_price):
        self.store_type = store_type
        self.store_nbr = store_nbr
        self.holiday_type = holiday_type
        self.onpromotion = onpromotion
        self.oil_price =oil_price


    def preprocessing_input(self, store_type, store_nbr, holiday_type, onpromotion, oil_price):
        # converting categoical columns into numerical columns
        store_type = STORE_TYPE[store_type]
        holiday_type = HOLIDAY_TYPE[holiday_type]

        # create a dataframe with user input 
        user_input = pd.DataFrame({
            'store_type': [store_type],
            'store_nbr': [store_nbr],
            'onpromotion': [onpromotion],
            'oil_price': [oil_price],
            'holiday_type': [holiday_type]
              })
        
        # perprocess the user input using the preprocessing
        preprocessed_input = preprocesser.transform(user_input)

        # return the preprocessed_ input data 
        return preprocessed_input

    # creating a prediction function
    def predict_price(self, preprocessed_input):

        # make prediction using the pre-trained model
        predicted_price = model.predict(preprocessed_input)

        # return the predicted sales of data
        return predicted_price[0]
    
    def predict_price_from_input(self):
        
        # preprocess the input using the preprocessor
        preprocessed_input = self.preprocessing_input(
            holiday_type = self.holiday_type, 
            oil_price = self.oil_price, 
            onpromotion = self.onpromotion, 
            store_nbr = self.store_nbr, 
            store_type = self.store_type
        )
        # make prediction using the pre-trained model
        predicted_price = self.predict_price(preprocessed_input)

         # Round off the predicted shipment price to two decimal places
        rounded_price = round(predicted_price, 2)
        predicted_price=rounded_price

        # Print the rounded predicted shipment price
        print("The predicted Sales is: $", predicted_price)
        
        return(predicted_price)
        
        
     