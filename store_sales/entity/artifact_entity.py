from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "ingestion_file_path",
    "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",[
    "schema_file_path",
    "report_file_path",
    "report_page_file_path",
    "validated_file_path",
    "message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",[
    "message",
    "transformed_train_file_path",
    "transformed_test_file_path",
    "time_series_data_file_path",
    "preprocessed_object_file_path",
    "feature_engineering_object_file_path"])

TimeDataTransformationArtifact = namedtuple("TimeDataTransformationArtifact",[
    "is_transformed",
    "message",
    "transformed_train_file_path",
    "transformed_test_file_path",
    "preprocessed_object_file_path",
    "feature_engineering_object_file_path"])




ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",[
    "is_trained",
    "message",
    "trained_model_object_file_path"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact",[
    "is_model_accepted",
    "improved_accuracy"])

ModelTrainerTIMEArtifact = namedtuple("ModelTrainerArtifact",[
    "message",
    "trained_model_object_file_path",
    "model_report",
    "prediction_image",
    "mse_score",
    "saved_report_file_path",
  "saved_model_file_path",
  "best_model_name",
    "saved_model_plot"])

ModelPusherArtifact=namedtuple("ModelPusherArtifact",["message"])