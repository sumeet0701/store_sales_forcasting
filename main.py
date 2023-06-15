from flask import Flask, render_template, request
from Prediction.batch_prediction import BatchPrediction
import pandas as pd
import io
from store_sales.pipeline.training_pipeline import Pipeline
from store_sales.logger import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Load the trained SARIMAX model
    model_file_path = r'C:\Users\Sumeet Maheshwari\Desktop\end to end project\store_sales_forcasting\store_sales_forcasting\Sales_Forecasting_Artifact\Artifact\model_training\2023-06-15-10-08-06\trained_time_model\model.pkl'  # Path to the trained model pickle file

    # Get the uploaded CSV file
    file = request.files['csv_file']
    if not file:
        return render_template('index.html', error='No CSV file uploaded.')

    # Read the CSV file
    try:
        data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        return render_template('index.html', error='Error reading CSV file: {}'.format(str(e)))

    # Extract the column names from the form
    exog_columns = ['onpromotion', 'holiday_type', 'family','store_type','store_nbr']
    target_column = 'sales'

    # Perform batch prediction
    batch_prediction = BatchPrediction(model_file_path)
    prediction_plot = batch_prediction.prediction(data, exog_columns, target_column)

    return render_template('index.html', prediction=prediction_plot)

@app.route('/train', methods=['POST'])
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

        return render_template('index.html', message="Training complete")

    except Exception as e:
        logging.error(f"{e}")
        error_message = str(e)
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)