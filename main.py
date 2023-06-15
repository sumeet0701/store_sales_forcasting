from flask import Flask, render_template, request
from Prediction.batch_predictions import batch_predictions
from store_sales.utils.utils import *
from store_sales.pipeline.training_pipeline import Pipeline
from store_sales.logger import logging
from store_sales.constant import *

import pandas as pd
import matplotlib.pyplot as plt

import io
import os, sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # load the training SARIMAX model
    model_file_path = r""

    #get the uploaded csv file
    file = request.file['csv_file']
    if not file:
        return render_template('index.html', error ='No csv file was uploaded')
    
    # read csv file
    try:
        logging.info('Reading csv file')
        data = pd.read_csv(io.StringIO(file.read().decode('utf8')))
    except Exception as e:
        return render_template('index.html', error ='Error reading CSV file: {}'.format(str(e)))

    # reading yaml file using utils helper functions
    time_series_config = read_yaml_file(file_path= TIME_CONFIG_FILE_PATH)
    exog_columns =time_series_config[EXOG_COLUMNS]
    traget_columns =time_series_config[TARGET_COLUMN]

    # dropping unwanted columns
    drop_col


                               
