from store_sales.exception import CustomException
from store_sales.logger import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import yaml
import os
import re
import sys


class LabelEncoderTransformer(TransformerMixin):

    def __init__(self,X,y=None):
        return self
    
    def transform(self, X):
        x_encoded = x.copy()
        for column in x_encoded.columns:
            x_encoded[column] = x_encoded[column].astype("category").cat.codes
            return x_encoded
    

def label_encoder_categorical_columns(data: pd.DataFrame, categorical_columns, traget_columns):
        # create a pipeline with the LabelEncoderTransformer
        pipeline = Pipeline([
            ("label_encoder", LabelEncoderTransformer())
        ])
        # appling label encoder to categorical columns in the input Dataframe
        df = data.copy()
        #Applying label encoder to categorical columns
        df_encoded = pipeline.fit_transform(df[categorical_columns])
        # combining encoded categorical columns with other columns in the input Dataframe
        df_combined = pd.concat([df_encoded, df.drop(categorical_columns), 
                                 df.drop(categorical_columns,axis = 1)], axis=1)

        #return df_combined
        return df_combined


class BatchPrediction:
    def __init__(self,model_file,data,
                 exog_columns, traget_columns,
                 drop_columns, label_columns,group_columns,sum_columns, mean_columns):
        # load the trained SARMIAX Model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

            # exog_columns
            self.exog_columns =exog_columns
            # traget_columns
            self.traget_columns =traget_columns
            # drop_columns
            self.drop_columns =drop_columns
            # label_column
            self.label_column =label_columns

            #group_column
            self.group_column =group_columns
            #sum_column
            self.sum_column =sum_columns
            # mean_column
            self.mean_column =mean_columns
    
    def get_model_name_from_yaml(self, file_path):
        """
        Extract the model name from a YAML file

        Args:
            file_path(str): The path to the YAML file
        Returns:
            str: The name of the model.
        """
        try:
            # reading yaml file
            with open(file_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # get the model name from the YAML data
            model_name = yaml_data['model_name']
            return model_name
        except  FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return None
        except Exception as e:
            raise CustomException(e,sys)
    
    def drop_columns(self, df, drop_columns):
        # list of columns to drop
        columns_to_drop = drop_columns

        # drop all the columns from dataframe
        df.drop(columns =columns_to_drop, inplace = True)

        drop_columns = ['year', 'month', 'week','quarter','day_of_week']

        # check if the columns exist
        exist_columns = [col for col in df.columns if col in df.columns]

        # droping existing columns
        df.drop(columns =exist_columns, inplace = True)

        # return clean dataframe
        return df
    
    def group_data(self, df,group_columns, sum_columns, mean_columns):

        """
        Groups the data in the DataFrame based on the specified group_columns,
        calculates the sum of sum_columns within each group, and calculates
        the mean of mean_columns within each group.
        
        Args:
            df (pandas.DataFrame): The input DataFrame.
            group_columns (list): A list of column names to group the data by.
            sum_columns (list): A list of column names to calculate the sum within each group.
            mean_columns (list): A list of column names to calculate the mean within each group.
            
        Returns:
            pandas.DataFrame: The modified DataFrame with group-wise sums and means.
        """
        # Group the data and calculate the sum of sum_columns within each group
        df_gp = df.groupby(group_columns)[sum_columns].sum()

        # calculate the mean of mean_columns within each group
        df_gp[mean_columns] = df.groupby(group_columns)[mean_columns].sum()
        
        #return grouped dataframe 
        return df_gp
    

    def sarima_predict(self, data):
        # assceing necessary data
        exog_columns = self.exog_columns
        traget_columns = self.traget_columns
        label_encode_columns = self.label_encode_columns

        df = data.copy()

        # setting Date columns as index 
        df['data'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        #droping unwanted columns
        df = self.drop_columns(df)
        #df.to_csv("After_drop.csv")

        # assuming you have a dataframe called 'df' with a column name 'Holiday_type' and
        df['holiday_type'] =df['holiday_type'].astype('category')

        #performing Label encoding on categorical columns
        # Perform label encoding on categorical columns
        df = label_encoder_categorical_columns(df,
                                               categorical_columns=label_encode_columns,
                                               target_column='sales')
        #df.to_csv("label_encode.csv")
  
        df_gp=self.group_data(df,
                              sum_columns=self.sum_column,
                              group_columns=self.group_column,
                              mean_columns=self.mean_column)
        df_gp.to_csv('grouped.csv')
        
        # Extract the time series data and exogenous variables
        time_series_data = df_gp[target_column]
        exog_data = df_gp[exog_columns]
        df_gp[target_column].to_csv('targest.csv')
        exog_data.to_csv('exog_data.csv')
        # Make predictions
        predictions = self.model.get_prediction(exog=exog_data)
        predicted_values = predictions.predicted_mean
        
        predicted_values.to_csv('predicted.csv')

        # Get the last few 100 values
        last_few_values = df_gp.iloc[-100:]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(last_few_values.index, last_few_values[target_column], label='Actual')
        plt.plot(last_few_values.index, predicted_values[-100:], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()

        # Create the batch prediction folder if it doesn't exist
        if not os.path.exists('batch_prediction'):
            os.makedirs('batch_prediction')

        # Save the plot in the batch prediction folder
        plot_file_path = os.path.join('batch_prediction', 'plot.png')
        plt.savefig(plot_file_path)
        plt.close()

        # Return the path to the plot file
        return plot_file_path
       
    
    def Prophet_predict(self,data):
        
        # Accessing necessary Data 
        exog_columns=self.exog_columns
        target_column=self.target_column
        
        drop_columns=self.drop_columns_list
        
        df = data.copy()

        # Setting Date column as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Dropping unnecessary columns
        df = self.drop_columns(df,drop_columns)

        # Renaming Date column
        df = df.rename(columns={'date': 'ds'})
       # df.to_csv("prophet_data.csv")

        # datatype --> category
        df = label_encode_categorical_columns(df,
                                              categorical_columns=self.label_encode_columns,
                                              target_column='sales')
        # Group data
        df_gp = self.group_data(df,
                                sum_columns=self.sum_column,
                                group_columns=self.group_column,
                                mean_columns=self.mean_column)
      #  df_gp.to_csv('grouped.csv')

        # Extract the time series data and exogenous variables
        time_series_data = df_gp[target_column]
        exog_data = df_gp[exog_columns]
        
       # exog_data.to_csv('exog_prophet.csv')

        # Prepare the input data for prediction
        df = df_gp.copy()
        df['ds'] = pd.to_datetime(df.index)
        df = df.rename(columns={'sales': 'y'})

        # Include exogenous variables
        if exog_columns is not None:
            for column in exog_columns:
                if column in df.columns:
                    df[column] = exog_data[column].values.astype(float)
                else:
                    raise ValueError(f"Column '{column}' not found in the input data.")

        # Make predictions
        predictions = self.model.predict(df)

        # Get the last few 100 values
        last_few_values = df_gp.iloc[-100:]

        # Get the corresponding predictions for the last 100 values
        last_few_predictions = predictions[predictions['ds'].isin(last_few_values.index)]
        

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(last_few_values.index, last_few_values[target_column], label='Actual')
        plt.plot(last_few_predictions['ds'].values, last_few_predictions['yhat'].values, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Prediction')
        plt.legend()

        
        # Create the batch prediction folder if it doesn't exist
        if not os.path.exists('batch_prediction'):
            os.makedirs('batch_prediction')

        # Save the plot in the batch prediction folder
        plot_file_path = os.path.join('batch_prediction', 'plot.png')
        
        plt.savefig(plot_file_path)
        plt.show()
        plt.close()
        
        # Round the values in the 'yhat' column to two decimal places
        last_few_predictions['yhat'] = last_few_predictions['yhat'].round(2)

        # Convert numpy array to DataFrame
        prediction_df = pd.DataFrame({'prediction': last_few_predictions['yhat'].values})

        # Save DataFrame as CSV
        prediction_csv = 'prediction.csv'
        prediction_path = os.path.join('batch_prediction', prediction_csv)
        prediction_df.to_csv(prediction_path, index=False)

        # Return the path to the plot file
        return plot_file_path
    def prediction(self,data):
        
        exog_columns=self.exog_columns
        target_column=self.target_column
        model_file = self.model
        name = self.get_model_name_from_yaml(file_path=r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\store_sales_forcasting\store_sales_forcasting\saved_model\model.pkl")
        
        if name:
            logging.info(f"Model Name :{name}")
            print(f"The model name is: {name}")

            # Check if the model is "sarima" or "prophet"
            if 'sarima' in name.lower():
                # Call Sarima_predict() method
                plot_file_path=self.Sarima_predict(data)
            elif 'prophet' in name.lower():
                # Call Prophet_predict() method
                plot_file_path=self.Prophet_predict(data)
            else:
                print("Unsupported model. Cannot perform prediction.")
                
        return plot_file_path