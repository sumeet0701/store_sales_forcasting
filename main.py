from flask import Flask, render_template, request
from Prediction.batch_prediction import BatchPrediction
from Prediction.instance_prediction import InstancePrediction

from store_sales.utils.utils import *
import pandas as pd
import io
from store_sales.pipeline.training_pipeline import Pipeline
from store_sales.logger import logging
from demo import demo
from store_sales.constant import *
import matplotlib.pyplot as plt
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Load the trained SARIMAX model
    model_file_path = r'C:\Users\Sumeet Maheshwari\Desktop\end to end project\store_sales_forcasting\store_sales_forcasting\saved_model\model.pkl'  # Path to the trained model pickle file

    # Get the uploaded CSV file
    file = request.files['csv_file']
    if not file:
        return render_template('index.html', error='No CSV file uploaded.')

    # Read the CSV file
    try:
        data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        return render_template('index.html', error='Error reading CSV file: {}'.format(str(e)))

    # Read Yaml 
    time_config=read_yaml_file(file_path=TIME_CONFIG_FILE_PATH)
    exog_columns=time_config[EXOG_COLUMNS]
    target_column=time_config[TARGET_COLUMN]
    
    # Drop columns 
    drop_columns=time_config[DROP_COLUMNS]
    
    # Label Encode columns
    label_encode=time_config[LABEL_ENCODE_COLUMNS]
    
    # Group column
    group_column=time_config[GROUP_COLUMN]
    sum_column=time_config[SUM_COLUMN]
    mean_column=time_config[MEAN_COLUMN]
    
    
    


    # Perform batch prediction
    batch_prediction = BatchPrediction(model_file_path,data, exog_columns, target_column,
                                       drop_columns,
                                       label_encode,
                                       group_column,
                                       sum_column,
                                       mean_column)
    prediction_plot = batch_prediction.prediction(data)
    plt.savefig(prediction_plot)
    
    

    return render_template('index.html', prediction=prediction_plot)

@app.route('/train', methods=['POST'])
def train():
    try:
        demo_pipeline = demo()
        demo_pipeline.main()

        return render_template('index.html', message="Training complete")

    except Exception as e:
        logging.error(f"{e}")
        error_message = str(e)
        return render_template('index.html', error=error_message)





if __name__ == '__main__':
    app.run(debug=True)