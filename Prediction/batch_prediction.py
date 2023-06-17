import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os 
import re
from store_sales.logger import logging
import yaml

class LabelEncoderTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        STORE_TYPE  = {'A':0, "B":1, "C":2, "D":3, "E":4}
        HOLIDAY_TYPE ={'Holiday':0, 'Event':1, 'Additional':2, 'Transfer':3, 'Work Day': 4, "Bridge":5}
        X_encoded['store_type'] = X_encoded['store_type'].map(STORE_TYPE)
        X_encoded['holiday_type'] = X_encoded['holiday_type'].map(HOLIDAY_TYPE)
        return X_encoded

def label_encode_categorical_columns(data: pd.DataFrame, categorical_columns, target_column):


    # Create the pipeline with the LabelEncoderTransformer
    pipeline = Pipeline([
        ('label_encoder', LabelEncoderTransformer())
    ])

    # Apply label encoding to categorical columns in the input DataFrame
    df = data.copy()
    # Apply label encoding to categorical columns
    df_encoded = pipeline.fit_transform(df[categorical_columns])

    # Combine encoded categorical columns with other columns
    df_combined = pd.concat([df_encoded, df.drop(categorical_columns, axis=1)], axis=1)

    return df_combined


class BatchPrediction:
    def __init__(self, model_file,data,exog_columns, target_column,
                drop_columns, label_encode, group_column, sum_column, mean_column):
        # Load the trained SARIMAX model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
                
            # Exog data 
            self.exog_columns=exog_columns
            
            # Target Columns 
            self.target_column=target_column
            
            # Drop columns 
            self.drop_columns_list=drop_columns
            
            # Label encode 
            self.label_encode_columns=label_encode
            
            
            # group 
            self.group_column=group_column
            self.sum_column=sum_column
            self.mean_column=mean_column
    def get_model_name_from_yaml(self,file_path):
        """
        Extracts the model name from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            str: The name of the model.
        """
        try:
            # Read the YAML file
            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)

            # Get the model name from the YAML data
            model_name = yaml_data['model_name']

            return model_name

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return None


    def drop_columns(self,df,drop_columns):
        # List of columns to drop
        columns_to_drop = drop_columns

        # Drop the columns from the DataFrame
        df = df.drop(columns=columns_to_drop)
        
        
       
        drop_columns = ['year', 'month', 'week', 'quarter', 'day_of_week']

        # Check if the columns exist
        existing_columns = [col for col in drop_columns if col in df.columns]

        # Drop the existing columns
        df.drop(existing_columns, axis=1, inplace=True)
    
        return df

    def group_data(self,df, group_columns, sum_columns, mean_columns):
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

        # Calculate the mean of mean_columns within each group
        df_gp[mean_columns] = df.groupby(group_columns)[mean_columns].mean()

        return df_gp
    def Sarima_predict(self, data):
        
        # Accessing necessary Data 
        exog_columns=self.exog_columns
        target_column=self.target_column
        # LAbel encode columns 
        label_encode_columns=self.label_encode_columns
        
        
        df=data.copy()
        
        # Setting Date column as index 
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Dropping unncessry columns 
        df=self.drop_columns(df)
        #df.to_csv("After_drop.csv")
        # Assuming you have a DataFrame called 'df' with a column named 'holiday_type'
        df['holiday_type'] = df['holiday_type'].astype('category')
        
        # Perform label encoding on categorical columns
        df = label_encode_categorical_columns(df,categorical_columns=label_encode_columns,target_column='sales')
        #df.to_csv("label_encode.csv")
  
        df_gp=self.group_data(df,
                              sum_columns=self.sum_column,
                              group_columns=self.group_column,
                              mean_columns=self.mean_column)
        df_gp = label_encode_categorical_columns(df_gp,categorical_columns=label_encode_columns,target_column='sales')
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
        # Group data
        df_gp = self.group_data(df,
                                sum_columns=self.sum_column,
                                group_columns=self.group_column,
                                mean_columns=self.mean_column)
      #  df_gp.to_csv('grouped.csv')
        df_gp = label_encode_categorical_columns(df_gp,categorical_columns=self.label_encode_columns,target_column='sales')

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
        name = self.get_model_name_from_yaml(file_path=r"E:\Ineuron may batch projects\Projects\Store sales Forecasting\saved_models\model.yaml")
        
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
    
    
    