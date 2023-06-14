from store_sales.exception import CustomException
from store_sales.logger import logging
from store_sales.utils.utils import read_yaml_file
from store_sales.utils.utils import save_array_to_directory
from store_sales.utils.utils import save_data,save_object
from store_sales.entity.config_entity import DataIngestionConfig
from store_sales.entity.config_entity import DataValidationConfig
from store_sales.entity.config_entity import DataTransformationConfig
from store_sales.entity.artifact_entity import DataIngestionArtifact
from store_sales.entity.artifact_entity import DataValidationArtifact
from store_sales.entity.artifact_entity import DataTransformationArtifact
from store_sales.constant import *

import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warning for setting with copy
# Set low_memory option to False
pd.options.mode.use_inf_as_na = False

import numpy as np
import sys 
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,numerical_columns,categorical_columns,
                 target_columns,drop_columns,date_column,
                 all_column,handling_categoical_columns,
                 time_series_data_path):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        

                                ############### Accesssing Column Labels #########################
                                
                                
                 #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #
                                
                                
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_columns = target_columns
        self.date_column=date_column
        self.columns_drop = drop_columns
        self.col=all_column
        self.handling_categoical_columns = handling_categoical_columns
        self.time_series_data_path=time_series_data_path

        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")
        
       ### Data Wrangling 
       
            ## Data Modification
    #           1. Dropping columns 
    #           2. 
    
    
    def drop_columns(self,df: pd.DataFrame):
        try:
            fe_drop = ['year', 'month', 'week', 'quarter', 'day_of_week',
            'locale','locale_name','description','city','state','transferred']
            
            columns_to_drop = [column for column in fe_drop if column in df.columns]
            columns_not_found = [column for column in fe_drop if column not in df.columns]

            if len(columns_not_found) > 0:
                logging.info(f"Columns not found: {columns_not_found}")
                return df

            logging.info(f"Dropping columns: {columns_to_drop}")
            df.drop(columns=columns_to_drop, axis=1, inplace=True)
            logging.info(f"Columns after dropping: {df.columns}")

            return df
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    def date_datatype(self,df:pd.DataFrame):
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        return df
      
    
    def convert_columns_to_category(self,df, columns):
        for col in columns:
            df[col] = df[col].astype('category')
            logging.info(f"Column '{col}' converted to 'category' data type.")
        return df
        
    
    def remove_special_chars_and_integers_from_unique_values(self,df, column_name):
        # Remove special characters and integers from unique values
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s:]', '', str(x)))
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\d+', '', str(x)))
        
        return df
    
    def replace_low_percentages(self,df:pd.DataFrame, column_name, threshold):
        # Calculate unique value percentages
        value_counts = df[column_name].value_counts()
        unique_value_percentages = (value_counts / len(df)) * 100

        # Identify unique values with percentage less than the threshold
        low_percentage_values = unique_value_percentages[unique_value_percentages < threshold].index

        # Replace low percentage values with 'Others'
        df[column_name].replace(low_percentage_values, 'Others', inplace=True)
        
        return df
    
    
    
    def check_duplicate_values(self,df):
        initial_shape = df.shape

        # Remove duplicates and get the modified DataFrame
        df_no_duplicates = df.drop_duplicates()

        modified_shape = df_no_duplicates.shape

        # Compute the count of duplicate values
        duplicated_count = initial_shape[0] - modified_shape[0]

        logging.info(f"Shape before removing duplicates: {initial_shape}")
        logging.info(f"Shape after removing duplicates: {modified_shape}")
        logging.info(f"Count of duplicate values: {duplicated_count}")

        return df_no_duplicates
    
    def renaming_oil_price(self,df: pd.DataFrame):
        df = df.rename(columns={"dcoilwtico": "oil_price"})
        
        logging.info(" Oil Price column renamed ")
        logging.info(f"{df.head()}")
        
        return df
    
    
    def missing_values_info(self,df: pd.DataFrame):
        if df.isnull().sum().sum() != 0:
            na_df = (df.isnull().sum() / len(df)) * 100
            na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
            
            logging.info("Missing ratio information:")
            for column, missing_ratio in na_df.iteritems():
                logging.info(f"{column}: {missing_ratio:.2f}%")
            
            logging.info(f"top 5 rows of data frame: \n {df.head()}")             
            
        else:
            logging.info("No missing values found in the dataframe.")
            
        return df    
    
    def drop_null_unwanted_columns(self,df:pd.DataFrame):
        if 'id' in df.columns:
            logging.info("Dropping 'id' column...")
            df.drop(columns=['id'], inplace=True)
        else:
            logging.info("'id' column not found. Skipping dropping operation.")

        logging.info("Dropping rows with null values...")
        #df.dropna(inplace=True)

        logging.info("Resetting DataFrame index...")
        #df.reset_index(drop=True, inplace=True)

        logging.info("Columns dropped, null values removed, and index reset.")
        logging.info(f"Top 5 rows of df: \n {df.head()}")

        return df
    
    def handling_missing_values(self,df):
        logging.info("Handling missing values in 'transactions' column...")
        df['transactions'] = df['transactions'].fillna(df['transactions'].mean())

        logging.info("Checking 'oil_price' column for missing values...")
        missing_values = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values: {missing_values}")

        logging.info("interpolate missing values in 'oil_price' column...")
        df['oil_price'].interpolate(method='linear', inplace=True)

                # Verify if missing values have been filled
        missing_values_after = df['oil_price'].isna().sum()
        logging.info(f"Number of missing values after filling: {missing_values_after}")

        columns_missing = ['holiday_type', 'locale', 'locale_name', 'description', 'transferred']
        logging.info(f"{df['holiday_type'].mode()}")

        for column in columns_missing:
            logging.info(f"Filling missing values in '{column}' column with mode...")
            if not df[column].empty:
                mode_value = df[column].mode().iloc[0]
                df[column].fillna(mode_value, inplace=True)
        logging.info("Missing values handled.")
        
        return df
    

    def remove_outliers_IQR(self,data, cols):
        for col in cols:
            logging.info(f"Removing outliers in column '{col}' using IQR method...")

            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outliers_removed = data[(data[col] > upper_limit) | (data[col] < lower_limit)]
            num_outliers_removed = outliers_removed.shape[0]

            logging.info(f"Number of outliers removed in column '{col}': {num_outliers_removed}")

            data[col] = np.where(data[col] > upper_limit, upper_limit,
                                np.where(data[col] < lower_limit, lower_limit, data[col]))

            logging.info(f"Column '{col}' modified: {num_outliers_removed} outliers modified")

        return data
            
            
    def map_categorical_values(self,df:pd.DataFrame):
        logging.info("Mapping unique values of categorical columns:")
        for column in df.select_dtypes(include=['category']):
            unique_values = df[column].unique()
            mapping = {value: i for i, value in enumerate(unique_values)}
            df[column] = df[column].map(mapping)
            logging.info(f"Column: '{column}', Unique Values: {unique_values}")
            
        return df
    
    def droping_unwanted_columns(self,df:pd.DataFrame):
        logging.info("Dropping unwanted columns form dataframe")
        columns = ['locale','locale_name','description','city','state','transferred']
        df = df.drop(columns=columns)
        return df

    
    def run_data_modification(self,df:pd.DataFrame):
          
        # Dropping Irrelevant Columns
        df= self.drop_columns(df)
        
        # Change Datatype of the column 
        df= self.date_datatype(df)
        
        # Set categorical Columns to category 
        #df=self.convert_columns_to_category(df,self.categorical_columns)
        
        # Removing special character from "Description"
        #df=self.remove_special_chars_and_integers_from_unique_values(df,'description')
        
        # Replace low percenatages unique values 
        #df=self.replace_low_percentages(df,'locale_name',0.5)
        #df=self.replace_low_percentages(df,'description',0.5)
        
        # renaming Oil_Price
        df=self.renaming_oil_price(df)
        
        # Drop Duplicated values 
        df=self.check_duplicate_values(df)
        
        # Missing Values info 
        df= self.missing_values_info(df)
        
        
        # dropping null values 
        df=self.drop_null_unwanted_columns(df)
        
        # handling missing Values 
        df=self.handling_missing_values(df)
        
        # Outlier column
        #outliers_mod_columns=['transactions']
        
        #df=self.remove_outliers_IQR(df,outliers_mod_columns)
        
        # Exported data
        # df.to_csv('removed_outliers.csv')
        
        # Rechecking datatypes 
        df=self.convert_columns_to_category(df,self.categorical_columns)

        # dropping unwanted columns
        df = self.droping_unwanted_columns(df)   
        # Saving data for time series training before map encoding
        time_series_data_path=os.path.join(self.time_series_data_path,TIME_SERIES_DATA_FILE_NAME)
        
        save_data(file_path = time_series_data_path, data = df)
   
        # Map Encoding 
        df=self.map_categorical_values(df)
        logging.info(f"{df.head()}")
       
        return df
       
       
       
       
    def data_wrangling(self,df:pd.DataFrame):
        try:

            
            # Data Modification 
            data_modified=self.run_data_modification(df)
            
            logging.info(" Data Modification Done")
            
            
            logging.info("Column Data Types:")
            for column in data_modified.columns:
                logging.info(f"Column: '{column}': {data_modified[column].dtype}")
            return data_modified
        except Exception as e:
            raise CustomException(e,sys) from e
         
    def fit(self,X,y=None):
            return self
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified = self.data_wrangling(X)
            col = self.col
            # Reindex the DataFrame columns according to the specified column sequence
            data_modified = data_modified.reindex(columns=col)

            #data_modified.to_csv("data_modified.csv", index=False)
            logging.info("Data Wrangling Done")
            arr = data_modified.values
                
            return arr
        except Exception as e:
            raise CustomException(e,sys) from e

class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
                                
                                
                                #           Schema.yaml -----> DataTransfomation 
            
            # Schema File path 
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            
            # Reading data in Schema 
            self.schema = read_yaml_file(file_path=self.schema_file_path)
            
            # Time series transaformed csv path 
            self.time_series_data_path=self.data_transformation_config.time_series_data_file_path
            
            # Column data accessed from Schema.yaml
            self.target_column_name = self.schema[TARGET_COLUMN_KEY]
            self.numerical_column_without_target=self.schema[NUMERICAL_COLUMN_WITHOUT_TAR]
            self.categorical_columns = self.schema[CATEGORICAL_COLUMN_KEY]
            self.date_column=self.schema[DATE_COLUMN]
            
            self.drop_columns=self.schema[DROP_COLUMN_KEY]
            self.handling_categoical_columns=self.schema[HANDLING_CATEGORICAL_COLUMN]
            self.col=self.numerical_column_without_target+self.categorical_columns+self.date_column+self.target_column_name
                                ########################################################################
        except Exception as e:
            raise CustomException(e,sys) from e


    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(numerical_columns=self.numerical_column_without_target,
                                                                            categorical_columns=self.categorical_columns,
                                                                            target_columns=self.target_column_name,
                                                                            date_column=self.date_column,
                                                                            all_column=self.col,
                                                                            drop_columns=self.drop_columns,
                                                                            handling_categoical_columns = self.handling_categoical_columns,
                                                                            time_series_data_path=self.time_series_data_path))])
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def split_train_test_data(self,input_df, output_df, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(input_df, output_df, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def get_data_transformer_object(self):
        try:

            logging.info('Creating Data Transformer Object')
            

            numerical_columns = self.numerical_column_without_target+self.handling_categoical_columns

            # Define transformers for numerical and categorical columns
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])


            # Combine the transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, numerical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) from e
        
    
    
    
    
        
    def initiate_data_transformation(self):
        try:
            
            logging.info(f"Obtaining file from file path ")
            validated_file_path = self.data_validation_artifact.validated_file_path

            logging.info(f"Loading Data as pandas dataframe.")
            validated_file_path=os.path.join(validated_file_path,"final_data.csv")
            file_data = pd.read_csv(validated_file_path)
            
            logging.info(f" Data columns {file_data.columns}")
            
            # Schema.yaml ---> Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns_without_target = self.numerical_column_without_target
            categorical_columns = self.categorical_columns
            handling_categorical_columns = self.handling_categoical_columns
            date_column=self.date_column
                        
            # Log column information
            logging.info("Numerical columns: {}".format(numerical_columns_without_target))
            logging.info("Categorical columns: {}".format(categorical_columns))
            logging.info("Target Column: {}".format(target_column_name))
            logging.info(f"Date column :{date_column}")
            
            
            col = self.col
            # All columns 
            logging.info("All columns: {}".format(col))
            
            
            # Reorder the columns in the DataFrame
            file_data = file_data.reindex(columns=col)

            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            #logging.info(f"Feature engineering object{fe_obj.shape}")

            logging.info(f"{type(fe_obj)}") 

            logging.info(f"Feature engineering object droping unwanted columns")
            #unwanted_columns = ['locale', 'locale_name', 'description','city', 'state','transferred']
            #fe_obj.drop()
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")

            feature_eng_arr = fe_obj.fit_transform(file_data)
            
            # Converting featured engineered array into dataframe
            logging.info(f"Converting featured engineered array into dataframe.")
            
            
            #logging.info(f"Columns for Feature Engineering : {col}")
            feature_eng_df = pd.DataFrame(feature_eng_arr,columns=col)
            logging.info(f"Feature Engineering - Train Completed")
            logging.info(f"{feature_eng_df.columns}")
            #feature_eng_df.to_csv('feature_eng_df.csv')
            
            
            # DataFrame
            target_column_name=target_column_name

            target_feature_df = feature_eng_df[target_column_name]
            input_feature_df = feature_eng_df.drop(columns = ['sales','date','locale', 'locale_name', 'description','city', 'state','transferred'],axis = 1)
            
            
            # Train and Test split 
            X_train, X_test, y_train, y_test = self.split_train_test_data(input_feature_df, target_feature_df, test_size= 0.2)
        
            
            logging.info(f" X_Train_Shape :{X_train.shape} Y_Train_Shape : {y_train.shape}")
            logging.info(f" X_Test_Shape :{X_test.shape} Y_Test_Shape : {y_test.shape}")
            
            logging.info(f"Input Columns in Split data : {X_train.columns}| Target Column {y_train.columns}")
            
            
            
            ## Preprocessing 
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            preprocessing_obj = self.get_data_transformer_object()
            
            # X_train and X_test
            train_arr = preprocessing_obj.fit_transform(X_train)
            test_arr = preprocessing_obj.transform(X_test)
        
            col=numerical_columns_without_target + handling_categorical_columns
            transformed_train_df = pd.DataFrame(np.c_[train_arr,np.array(y_train)],columns=col+target_column_name)
            transformed_test_df = pd.DataFrame(np.c_[test_arr,np.array(y_test)],columns=col+target_column_name)
           
           # Transformed Train and Test path 
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir  
            
            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")
            
            transformed_test_file_path = os.path.join(transformed_test_dir,"transformed_test.csv")
        
            
            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test file")
            
            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            save_data(file_path = transformed_test_file_path, data = transformed_test_df)
            logging.info("Transformed Train and Transformed test file saved")
            
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)
            
            
            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)
            
            time_series_data_file_path=os.path.join(self.time_series_data_path,TIME_SERIES_DATA_FILE_NAME)
            
            data_transformation_artifact = DataTransformationArtifact(
            message="Data transformation successfull.",
            transformed_train_file_path = transformed_train_file_path,
            transformed_test_file_path = transformed_test_file_path,
            time_series_data_file_path=time_series_data_file_path,
            preprocessed_object_file_path = preprocessing_object_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")