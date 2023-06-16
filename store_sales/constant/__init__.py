import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_KEY = "config"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

TIME_CONFIG_FILE_NAME='time_series.yaml'
TIME_CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TIME_CONFIG_FILE_NAME)

EXOG_COLUMNS='exog_columns'

TARGET_COLUMN='target_column'
LABEL_ENCODE_COLUMNS='label_encode'

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline"

SCHEMA_CONFIG_KEY='schema_config'
SCHEMA_DIR_KEY ='schema_dir'
SCHEMA_FILE_NAME='schema_file'

TARGET_COLUMN_KEY='target_column'
NUMERICAL_COLUMN_KEY='numerical_columns'
NUMERICAL_COLUMN_WITHOUT_TAR='numerical_columns_without_target'
CATEGORICAL_COLUMN_KEY='categorical_columns'
HANDLING_CATEGORICAL_COLUMN = 'handling_categorical_columns'
DROP_COLUMN_KEY='drop_columns'
DATE_COLUMN='date_columns'
LABEL_ENCODER_COLUMNS = 'label_encoder_columns'



# Data Ingestion related variables
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"



FILE_PATH="file_path"
FILE_NAME="file_name"
DESTINATION_FOLDER= r"C:\Users\Sumeet Maheshwari\Desktop\end to end project\store_sales_forcasting\store_sales_forcasting\Jup_Notebook\Merged_dataset\cleaned_data"

# Database related variables
DATABASE_CLIENT_URL_KEY = "mongodb://localhost:27017/?readPreference=primary&ssl=false&directConnection=true"
DATABASE_NAME_KEY = "Store_Sales_db"
DATABASE_COLLECTION_NAME_KEY = "store_sales"
DATABASE_TRAINING_COLLECTION_NAME_KEY = "Training"
DATABASE_TEST_COLLECTION_NAME_KEY = "Test"

# Data Validation related variable
DATA_VALIDATION_ARTIFACT_DIR="data_validation"
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_VALID_DATASET ="validated_data"
DATA_VALIDATION_TRAIN_FILE = "Train_data"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"


# Data Transformation related variables
PIKLE_FOLDER_NAME_KEY='Prediction_Files'
# Artifact
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"

# key  ---> config.yaml---->values
# Data Transformation related variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY ="feature_engineering_object_file_name"
DATA_TRANSFORMATION_TIME_SERIES_DATA_DIR='time_series_data'


TIME_SERIES_DATA_FILE_NAME='time_model_file_name'


PIKLE_FOLDER_NAME_KEY = "prediction_files"

# Model Training related variables
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_ARTIFACT_DIR = "model_training"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"

# saved Model Directory 
SAVED_MODEL_DIRECTORY='saved_models'
MODEL_FILE_NAME='model.pkl'
MODEL_REPORT_FILE='model.yaml'


PIKLE_FOLDER_NAME_KEY = "prediction_files"

# Model Training related variables
MODEL_TRAINER_TIME_CONFIG_KEY = "model_trainer_time_config"
MODEL_TRAINER_ARTIFACT_DIR = "model_training"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_REPORT_FILE_NAME='model_report_file_name'
PREDICTION_IMAGE='prediction_image'
MODEL_TRAINER_FILE = "time_model_file_name"


GROUP_COLUMN ='group_columns'
SUM_COLUMN = 'sum_columns'
MEAN_COLUMN = 'mean_columns'

DROP_COLUMNS='drop_columns'

# Prediction 
PREDICTION_ROWS='prediction_rows'