from collections import namedtuple


DataIngestionConfig=namedtuple("DataIngestionConfig",[
    "raw_data_dir",
    "ingested_data_dir"
    ])



DataValidationConfig = namedtuple("DataValidationConfig",[
    "schema_file_path",
    "file_path",
    "report_file_path",
    "report_page_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig",[
    "transformed_train_dir",
    "transformed_test_dir",
    "time_series_data_file_path",
    "preprocessed_object_file_path",
    "feature_engineering_object_file_path"])

TimeDataTransformationConfig = namedtuple("TimeDataTransformationConfig",[
    "time_transformed_train_dir",
    "time_transformed_test_dir",
    "time_preprocessed_object_file_path",
    "time_feature_engineering_object_file_path"])

DatabaseConfig = namedtuple("DatabaseConfig",[
    "client_url",
    "database_name",
    "collection_name",
    "training_collection_name",
    "test_collection_name"])


ModelTrainerConfig = namedtuple("ModelTrainerConfig",["trained_model_file_path"])



TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",["artifact_dir"])

ModelTrainerTIMEConfig = namedtuple("ModelTrainerConfig",[
    "trained_model_file_path",
     "time_Series_grouped_data",
     "model_report",
     "prediction_image",
     "best_model_png",
     "saved_model_file_path",
     "saved_report_file_path",
     "saved_model_plot"])
