import shutil
import pickle
from store_sales.entity.config_entity import ModelTrainerTIMEConfig
from store_sales.entity.artifact_entity import DataTransformationArtifact
from store_sales.entity.artifact_entity import ModelTrainerArtifact,ModelTrainerTIMEArtifact
from store_sales.constant import *
import sys
import os
from store_sales.logger import logging
from store_sales.exception import CustomException
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from store_sales.utils.utils import read_yaml_file,save_image,save_object
import matplotlib.pyplot as plt
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from prophet import Prophet
import numpy as np
import re   
import yaml

class LabelEncoderTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column in X_encoded.columns:
            X_encoded[column] = X_encoded[column].astype('category').cat.codes
        return X_encoded

def label_encode_categorical_columns(df:pd.DataFrame, categorical_columns):
    # Create the pipeline with the LabelEncoderTransformer
    pipeline = Pipeline([
        ('label_encoder', LabelEncoderTransformer())
    ])

    # Apply label encoding to categorical columns
    df_encoded = pipeline.fit_transform(df[categorical_columns])

    # Combine encoded categorical columns with other columns
    df_combined = pd.concat([df_encoded, df.drop(categorical_columns, axis=1)], axis=1)

    df_encoded = df_combined.copy()
    return df_encoded

import pmdarima as pm



import pandas as pd

def group_data(df, group_columns, sum_columns, mean_columns):
    # Group by the specified columns and calculate the sum and mean
    df = df.groupby(group_columns)[sum_columns].sum()
    
    df[mean_columns] = df.groupby(group_columns)[mean_columns].mean()


    return df



class SarimaModelTrainer:
    def __init__(self, model_report_path,target_column,exog_columns,image_directory):
        self.model_report_path = model_report_path
        self.target_column=target_column
        self.exog_columns=exog_columns
        self.image_directory=image_directory

    def fit_auto_arima(self, df, target_column, exog_columns=None, start_p=2, start_q=0,
                       max_p=3, max_q=2, m=7, start_P=0, seasonal=True, d=0, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True,
                       stepwise=True):

        data = df[target_column]
        exog = df[exog_columns] 
        
        logging.info(" Starting auto arima ......")
        model = pm.auto_arima(data, exogenous=exog, start_p=start_p, start_q=start_q,
                              max_p=max_p, max_q=max_q, m=m, start_P=start_P,
                              seasonal=seasonal, d=d, D=D, trace=trace,
                              error_action=error_action, suppress_warnings=suppress_warnings,
                              stepwise=stepwise)

        order = model.order
        seasonal_order = model.seasonal_order
        print('order:', order)
        print('seasonal order:', seasonal_order)

        return order, seasonal_order

    def fit_sarima(self, df, target_column, exog_columns=None, order=None, seasonal_order=None, trend='c'):
        # Fit SARIMA model based on helper plots and print the summary.
        data = df[target_column]
        exog_columns=['onpromotion','holiday_type','oil_price']
        exog = df[exog_columns]
        
        logging.info(f" Exog Columns : {exog.columns}")

        sarima_model = SARIMAX(data, exog=exog,
                               order=order,
                               seasonal_order=seasonal_order,
                               trend=trend).fit()
        return sarima_model

    def get_sarima_summary(self, model, file_location):
        # Write the SARIMA model summary in YAML format to the specified file
        summary = model.summary()
        summary_yaml = yaml.dump(summary._tables[0].as_map(), default_flow_style=False)

        with open(file_location, 'w') as file:
            file.write(summary_yaml)

        print(f"Summary written to {file_location}")


    def forecast_and_predict(self, df, target_column, model, exog_columns=None, num_days=70):
        last_60_days = df.iloc[-num_days:]
        exog_columns = ['onpromotion', 'holiday_type', 'oil_price']
        # Extract the exogenous variables for the last 60 days
        exog_data = last_60_days[exog_columns]

        forecast = model.get_prediction(start=last_60_days.index[0], end=last_60_days.index[-1], exog=exog_data)
        predicted_values = forecast.predicted_mean

        # Calculating residuals
        actual_values = df[target_column].values[-num_days:]
        predicted_values_numpy = predicted_values.values  # Convert predicted_values to numpy array
        residuals = actual_values - predicted_values_numpy

        # Calculate mean squared error
        mse = np.mean(residuals ** 2)

        return predicted_values, mse
    
    def save_image(self,df, target_column, predicted_values, num_days=70, image_name='Sarima_exog.png'):
        # Plotting actual and predicted values for the last few rows
        plt.plot(df[target_column].tail(num_days), label='Actual')
        plt.plot(predicted_values.tail(num_days).index, predicted_values.tail(num_days), label='Forecast')
        plt.legend()

        # Rotate x-axis labels by 90 degrees
        plt.xticks(rotation=90)

        # Save the plot as an image
        os.makedirs(self.image_directory,exist_ok=True)
        plot_image_path = os.path.join(self.image_directory, image_name)
        plt.savefig(plot_image_path)

        # Close the plot to release memory
        plt.close()

        return plot_image_path

    def train_model(self, df):
        
        # Accessing column Labels 
        target_column=self.target_column
        exog_columns=self.exog_columns
        
        logging.info("Model Training Started: SARIMAX with EXOG data")

        # Perform auto ARIMA to get the best parameters
        #order, seasonal_order = self.fit_auto_arima(df, target_column, exog_columns)
        order=(2, 0, 1)
        seasonal_order=(0, 1, 1, 7)
        logging.info("Model trained best Parameters:")
        logging.info(f"Order: {order}")
        logging.info(f"Seasonal order: {seasonal_order}")

        # Fit the SARIMA model
        Sarima_model_fit = self.fit_sarima(df, target_column, exog_columns, order, seasonal_order, trend='c')

        # Dump summary in the report
        #self.get_sarima_summary(model, self.model_report_path)

        # Save prediction image and get predicted values and residuals
        predicted_values, mse = self.forecast_and_predict(df, target_column, Sarima_model_fit, exog_columns)
        
        # Plot and save Image of Forecast 
        plot_image_path=self.save_image(df, target_column, predicted_values, num_days=70, image_name='Sarima_exog.png')
        
        
        
        return  mse,Sarima_model_fit,plot_image_path



class ProphetModelTrainer:
    def __init__(self,target_column):
        
        self.target_column=target_column
        pass

    def fit_prophet(self, df, target_column, test_size=0.2):
        # Rename index to 'ds' and target column to 'y'
        df = df.rename(columns={target_column: 'y'}).reset_index().rename(columns={'date': 'ds'})

        # Perform train-test split
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

        # Fit Prophet model based on the train data
        prophet_model = Prophet()
        prophet_model.fit(train_df)

        self.model = prophet_model
        self.test_data = test_df

        print(prophet_model.params)

    def plot_actual_vs_predicted(self, last_n_days=100):
        if self.model is None or self.test_data is None:
            raise ValueError("Prophet model is not fitted. Call 'fit_prophet_with_plots' first.")

        # Make predictions on the test data
        forecast = self.model.predict(self.test_data)
        predicted_values = forecast['yhat'].values

        # Get the actual values from the test data
        actual_values = self.test_data['y'].values

        # Plot actual and predicted values for the last n days
        plt.plot(actual_values[-last_n_days:], label='Actual')
        plt.plot(predicted_values[-last_n_days:], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.title('Actual vs. Predicted Sales (Last {} Days)'.format(last_n_days))
        plt.legend()
        plt.show()

    def forecast_and_plot(self):
        if self.model is None or self.test_data is None:
            raise ValueError("Prophet model is not fitted. Call 'fit_prophet_with_plots' first.")

        forecast_and_plot(self.model, self.test_data)

import matplotlib.pyplot as plt
from prophet import Prophet

class ProphetModel_Exog:
    def __init__(self, exog_columns,image_directory):
        self.exog_columns = exog_columns,
        self.image_directory=image_directory
    
    def prepare_data(self, df: pd.DataFrame):

        df = df.rename_axis('ds').reset_index()
        df = df.rename(columns={'sales': 'y'})
        # Set the 'date' column as the index
        # Reset the index and rename columns
        

        exog_columns = self.exog_columns
        exog_columns=['onpromotion','holiday_type','oil_price','store_type','store_nbr']
        # Select the desired columns
        df = df[['ds','y','onpromotion','holiday_type','oil_price','store_type','store_nbr']]
        
        df.to_csv('prepare_Data.csv')

        return df

    def fit_prophet_model(self, data:pd.DataFrame):
        # Initialize Prophet model
        data=data
        m = Prophet()
        
        

        exog_columns=self.exog_columns
        logging.info(f" Adding exlog columns to the model : {exog_columns}")
        exog_columns=['onpromotion','holiday_type','oil_price','store_type','store_nbr']
        # Add exogenous regressors
        for column in exog_columns:
            m.add_regressor(column)

        # Fit the model with data
        m.fit(data)
        data.to_csv('after_fit.csv')
        
        logging.info(f" Data fit Prophet_Exog_data with columns : {data.columns}")

        return m,data

    def make_prophet_prediction(self, model, data):
        # Create future dataframe
        future = model.make_future_dataframe(periods=0)
        exog_columns=['onpromotion','holiday_type','oil_price','store_type','store_nbr']
        # Add exogenous variables to the future dataframe
        for column in exog_columns:
            future[column] = data[column]
            
        #future.to_csv('dataframe_will_prediction.csv')

        # Make prediction
        forecast_df = model.predict(future)

        return forecast_df

    def save_forecast_plot(self, forecast_df, actual_df):
        # Select the necessary columns for forecast and actual values
        forecast_tail = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)
        actual_values = actual_df['y'].tail(60)

        # Plot the forecasted values and actual values
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_tail['ds'], forecast_tail['yhat'], label='Forecast')
        plt.plot(forecast_tail['ds'], actual_values, label='Actual')
        plt.fill_between(forecast_tail['ds'], forecast_tail['yhat_lower'], forecast_tail['yhat_upper'],
                        alpha=0.3, label='Confidence Interval')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Forecasted Values and Actual Values with Confidence Interval')
        plt.legend()

        # Save the plot as an image
        image_name = 'prophet_exog.png'  # Change the file name and path as desired
        plot_image_path = os.path.join(self.image_directory, image_name)
        plt.savefig(plot_image_path)
        plt.close()  # Close the plot to free up memory

        # Calculate mean squared error
        mse = np.mean((actual_values - forecast_tail['yhat'])**2)

        return mse,plot_image_path
        
    def run_prophet_with_exog(self,df:pd.DataFrame):
        
        exog_columns=self.exog_columns
        # Prepare the data
        data = self.prepare_data(df=df)

        # Fit the Prophet model
        model,data_fit = self.fit_prophet_model(data)

        # Make predictions
        forecast_df = self.make_prophet_prediction(model,data)
        
        
        mse,plot_image_path=self.save_forecast_plot(forecast_df,data)
        
  

        # Return the image path
        return mse,model,plot_image_path

 
        
    

class ModelTrainer_time:
    def __init__(self, 
                 model_trainer_config: ModelTrainerTIMEConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'*'*20} Model Training started {'*'*20}\n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
            # Image save file location 
            self.image_directory=self.model_trainer_config.prediction_image
            
            #
            
            
            
            # Accessing Model report path 
            self.model_report_path=self.model_trainer_config.model_report
            
            # Time config.yaml 
            self.time_config_data= read_yaml_file(file_path=TIME_CONFIG_FILE_PATH)
                # Accessing columns
            self.exog_columns=self.time_config_data[EXOG_COLUMNS]
            self.target_column=self.time_config_data[TARGET_COLUMN]
            
            
           # Label encoding columns 
            self.label_encoding_columns=self.time_config_data[LABEL_ENCODE_COLUMNS]
            
            # Grouping columns 
            self.group_column=self.time_config_data[GROUP_COLUMN]
            self.sum_column =  self.time_config_data[SUM_COLUMN]
            self.mean_column=self.time_config_data[MEAN_COLUMN]
            
            # Dropping columns 
            self.drop_columns=self.time_config_data[DROP_COLUMNS]
            
            # Saved file paths 
            self.saved_model_path=self.model_trainer_config.saved_model_file_path
            self.saved_model_report_path=self.model_trainer_config.saved_report_file_path
            self.saved_model_plot=self.model_trainer_config.saved_model_plot
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def select_best_model(self,model1, model2, model1_mse, model2_mse, prediction_plot1, prediction_plot2):
        # Compare the MSE values
        model1_mse = float(np.mean(model1_mse))
        model2_mse = float(np.mean(model2_mse))

        if model1_mse < model2_mse:
            best_model = model1
            best_model_mse = model1_mse
            best_prediction_plot = prediction_plot1
        else:
            best_model = model2
            best_model_mse = model2_mse
            best_prediction_plot = prediction_plot2

        return best_model, best_model_mse, best_prediction_plot  
    
    def dump_model_info(self, mse, model, file_path):
        model_name = re.search(r"([^.]+)$", model.__class__.__name__).group(1)
        mse_float = float(np.mean(mse))
        model_info = {
            'mse': mse_float,
            'model_name': model_name
        }

        # Save the model as a pickle object
        model_file_path=self.model_trainer_config.trained_model_file_path
        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
        
        # Save the model info as YAML
        model_report_file_path = self.model_report_path
        with open(model_report_file_path, 'w') as file:
            yaml.dump(model_info, file)

        return model_report_file_path
    def copy_image(self,source_path, destination_path):
        # Extract the file name from the source path
        file_name = os.path.basename(source_path)

        # Construct the destination path with the file name
        destination_file_path = os.path.join(destination_path, file_name)

        # Copy the image file to the destination directory
        shutil.copyfile(source_path, destination_file_path)

        return destination_file_path
    
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding Feature engineered data ")
            Data_file_path=self.data_transformation_artifact.time_series_data_file_path


            logging.info("Accessing Feature Trained csv")
            data_df:pd.DataFrame= pd.read_csv(Data_file_path)
            
            data_df.to_csv('before_time_training.csv')
            
            target_column_name = 'sales'
           # logging.info("Splitting Input features and Target Feature")
            #target_feature = data_df[target_column_name]
          #  input_feature = data_df.drop(columns=[target_column_name], axis=1)
            
            
            # Setting Date column as index 
            data_df['date'] = pd.to_datetime(data_df['date'])
            data_df.set_index('date', inplace=True)
            
            # Dropping unncessry columns 
            drop_columns=self.drop_columns
            data_df.drop(drop_columns,axis=1,inplace=True)
            
            logging.info(f" Columns Dropped : ")
            
            logging.info(f" Columns : {data_df.columns}")

            
            # Label Encode categorical columns 
            #categorical_columns=['store_type', 'store_nbr','onpromotion']
            df=label_encode_categorical_columns(data_df,categorical_columns=self.label_encoding_columns)
        
            # Grouping data 
            group_columns = self.group_column
            sum_columns = self.sum_column
            mean_columns = self.mean_column
            
            # Data used for time series prediciton 
            os.makedirs(self.model_trainer_config.time_Series_grouped_data,exist_ok=True)
            grouped_data_file_path =os.path.join(self.model_trainer_config.time_Series_grouped_data,self.time_config_data[TIME_SERIES_DATA_FILE_NAME])
            
            #df_gp=group_data(df, group_columns, sum_columns, mean_columns)
            df_gp = df.groupby(group_columns)[sum_columns].sum()

            # Calculate the mean of 'oil_price' within each date group
            df_gp[mean_columns] = df.groupby(group_columns)[mean_columns].mean()
            
            df_gp.to_csv(grouped_data_file_path)
            
            
            # Training SARIMA MODEL 
            logging.info("-----------------------------")
            image_directory=self.image_directory=self.model_trainer_config.prediction_image
            os.makedirs(image_directory,exist_ok=True)
            logging.info("Starting SARIMA Model Training")
            sarima_model=SarimaModelTrainer(model_report_path=self.model_report_path,
                                            target_column=self.target_column,
                                           exog_columns=self.exog_columns,
                                            image_directory=self.image_directory)
            mse_Sarima,Sarima_exog_model,plot_image_path_sarima=sarima_model.train_model(df_gp)
            
            logging.info(" Sarima Model training completed")

            
            logging.info(f" Mean Sqaured Error :{mse_Sarima}")
            
            
             #Training Prophet - without exog 
            
            
            
            # Training Prophet - with exog data 
            logging.info("-----------------------------")
            image_directory=self.image_directory
            os.makedirs(image_directory,exist_ok=True)
            logging.info("Starting Prophet Model Training")
            
            prophet_exog=ProphetModel_Exog(exog_columns=self.exog_columns,image_directory=self.image_directory)
            mse_prophet_exog,Prophet_exog_model,plot_image_path_prohet_exog=prophet_exog.run_prophet_with_exog(df_gp)
            
            logging.info(" Prophet_Exog Model training completed")

            
            logging.info(f" Mean Sqaured Error :{mse_prophet_exog}")
            logging.info("Prophet training completed")
            
            best_model,mse_score,best_plot=self.select_best_model(Sarima_exog_model,Prophet_exog_model,
                                                        mse_Sarima,mse_prophet_exog,
                                                        plot_image_path_sarima,plot_image_path_prohet_exog)
            
            
            # Best plot location  
            logging.info(f"best image plot location : {best_plot}")
            prediction_image=self.model_trainer_config.best_model_png
            self.copy_image(source_path=best_plot,destination_path=prediction_image)
            
            # Saving Model
            trained_model_object_file_path=self.model_trainer_config.trained_model_file_path
            save_object(file_path=trained_model_object_file_path,obj=best_model)
            
            # Saving Report - Best model 
            model_report_path=self.dump_model_info(model=best_model,mse=mse_score,file_path=self.model_report_path)
            
            # Model name 
            best_model_name = re.search(r"([^.]+)$", best_model.__class__.__name__).group(1)
            
            model_trainer_artifact = ModelTrainerTIMEArtifact(model_report=model_report_path,
                                                            prediction_image=prediction_image,
                                                            message="Model_Training_Done!!",
                                                            trained_model_object_file_path=trained_model_object_file_path,
                                                            saved_report_file_path=self.saved_model_report_path,
                                                            saved_model_file_path=self.saved_model_path,
                                                            mse_score=float(np.mean(mse_score)),
                                                            best_model_name=best_model_name,
                                                            saved_model_plot=self.saved_model_plot)
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact



        except Exception as e:
            raise CustomException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Model Training log completed {'*'*20}\n\n")