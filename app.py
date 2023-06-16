from flask import Flask, render_template,request, send_file, redirect,url_for,flash
from flask_cors import CORS, cross_origin
from Prediction.batch_prediction import BatchPrediction
from Prediction.instance_prediction import InstancePrediction
from store_sales.pipeline.training_pipeline import Pipeline
from store_sales.constant import *
from store_sales.logger import logging
import shutil


input_file_path = "outier_clean.csv"
feature_engineering_file_path = "prediction_files/feat_eng.pkl"
transformer_file_path = 'prediction_files/preprocessed.pkl'
model_file_path = "saved_model/model.pkl"


app = Flask(__name__, template_folder = 'templates')
CORS(app)
app.secret_key = APP_SECRET_KEY

@app.route('/', methods = ['GET'])
@cross_origin()
def home():
    return render_template('result.html')


@app.route('/singe_prediction', methods  =['POST'])
@cross_origin()
def single_prediction():
    try:
        data ={
            "store_nbr": int(request.form['Store Number']),
            "store_type": request.form['store_type'],
            "onpromotion": int(request.form['OnPromotion']),
            "holiday_type": request.form['holiday_type'],
            "oil_price": int(request.form['oil_price'])
        }
        pred = InstancePrediction()
        preprocess  = pred.preprocessing_input(data =data)
        output = pred.predict_price(preprocess)
        flash(f"Predicted Cost for Shipment for given conditions: {output}","success")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))
    




if __name__=="__main__":
    
    port = int(os.getenv("PORT",5000))
    host = '0.0.0.0'
    app.run(host=host,port=port,debug=True)