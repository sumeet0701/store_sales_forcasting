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
        for column in X_encoded.columns:
            X_encoded[column] = X_encoded[column].astype('category').cat.codes
        return X_encoded


def label_encode_categorical_columns(data: pd.DataFrame, categorical_columns, target_column):
    # Split the data into target and input DataFrames
    

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
    def __init__(self, model_file):
        # Load the trained SARIMAX model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
            
            # group 
            self.group_column=['date']
            self.sum_column=['onpromotion','sales','holiday_type',"store_type","store_nbr"]
            self.mean_column=['oil_price']
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


    def drop_columns(self,df):
        # List of columns to drop
        columns_to_drop = [
            "family",
            "locale",
            "locale_name",
            "description",
            "city",
            "state",
            "cluster",
            "transactions",
            "transferred",
            "year",
            "month",
            "week",
            "quarter",
            "day_of_week"
        ]

        # Drop the columns from the DataFrame
        df = df.drop(columns=columns_to_drop)
    
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
    def Sarima_predict(self, data, exog_columns, target_column):
        
        df=data.copy()
        
        # Setting Date column as index 
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Dropping unncessry columns 
        df=self.drop_columns(df)
        #df.to_csv("After_drop.csv")
        # Assuming you have a DataFrame called 'df' with a column named 'holiday_type'
        df['holiday_type'] = df['holiday_type'].astype('category')
        df["store_type"] = df['store_type'].astype('category')
        # Perform label encoding on categorical columns
        df = label_encode_categorical_columns(df,categorical_columns=['holiday_type','onpromotion','store_type'],target_column='sales')
        #df.to_csv("label_encode.csv")
        # group data 
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
       
    
    def Prophet_predict(self, data, exog_columns, target_column):
        df = data.copy()

        # Setting Date column as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Dropping unnecessary columns
        df = self.drop_columns(df)

        # Renaming Date column
        df = df.rename(columns={'date': 'ds'})
        df.to_csv("prophet_data.csv")

        # datatype --> category
        df['holiday_type'] = df['holiday_type'].astype('category')
        df = label_encode_categorical_columns(df,categorical_columns=['holiday_type','onpromotion','store_type'],target_column='sales')
        # Group data
        df_gp = self.group_data(df,
                                sum_columns=self.sum_column,
                                group_columns=self.group_column,
                                mean_columns=self.mean_column)
        df_gp.to_csv('grouped.csv')

        # Extract the time series data and exogenous variables
        time_series_data = df_gp[target_column]
        exog_data = df_gp[exog_columns]
        
        exog_data.to_csv('exog_prophet.csv')

        # Prepare the input data for prediction
        df = df_gp.copy()
        df['ds'] = pd.to_datetime(df.index)
        df = df.rename(columns={target_column: 'y'})

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

        plt.show()
        # Create the batch prediction folder if it doesn't exist
        if not os.path.exists('batch_prediction'):
            os.makedirs('batch_prediction')

        # Save the plot in the batch prediction folder
        plot_file_path = os.path.join('batch_prediction', 'plot.png')
        plt.savefig(plot_file_path)
        plt.close()

        # Return the path to the plot file
        return plot_file_path
    
    def prediction(self,data, exog_columns, target_column):
        
        model_file = self.model
        name = self.get_model_name_from_yaml(file_path=r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\Store Sales Forcasting using Time series\store-sales-forcasting\saved_models\model.yaml")
        
        if name:
            logging.info(f"Model Name :{name}")
            print(f"The model name is: {name}")

            # Check if the model is "sarima" or "prophet"
            if 'sarima' in name.lower():
                # Call Sarima_predict() method
                plot_file_path=self.Sarima_predict(data, exog_columns, target_column)
            elif 'prophet' in name.lower():
                # Call Prophet_predict() method
                plot_file_path=self.Prophet_predict(data, exog_columns, target_column)
            else:
                print("Unsupported model. Cannot perform prediction.")
                
        return plot_file_path