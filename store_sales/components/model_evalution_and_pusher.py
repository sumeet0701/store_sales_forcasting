from store_sales.exception import CustomException
from store_sales.logger import logging
import sys
import os
from store_sales.entity.config_entity import *
from store_sales.entity.artifact_entity import *
from store_sales.utils.utils import load_pickle_object
from store_sales.utils.utils import save_object
from store_sales.utils.utils import save_pickle_object
from store_sales.utils.utils import copy_image
from store_sales.constant import *
import yaml
import numpy as np
class Model_eval_and_Pusher:

    def read_model_info(self,file_path):
        with open(file_path, 'r', encoding='latin1') as file:
            model_info = yaml.safe_load(file)
        
        model_name = model_info.get('model_name')
        mse_score = model_info.get('mse')
        
        return model_name, mse_score
    
    
    def save_model_report(self,mse, model_name, model_report_path):
        logging.info(f" Saving report at : {model_report_path}")

        model_name = model_name
        mse_float = float(np.mean(mse))
        logging.info(f" Model Name : {model_name}")
        logging.info(f" MSE  : {mse_float}")
        
        model_info = {
            'mse': mse_float,
            'model_name': model_name
        }
        logging.info(f"Dumping Data {model_info}")
        # Saving model report to a file
        with open(model_report_path, 'w') as file:
            yaml.dump(model_info, file)


    def compare_models(self,saved_model_report_path, model_trained_report, saved_model_path, model_trained_artifact_path):
        # Reading data from the saved model report
        logging.info(f" Accessing saved model report :{saved_model_report_path}")
        saved_model_name, saved_mse_score = self.read_model_info(file_path=saved_model_report_path)
        logging.info(f" Accessing Artifact  model report :{model_trained_report}")
        # Reading data from the artifact model report
        artifact_model_name, artifact_model_mse = self.read_model_info(file_path=model_trained_report)
        
        # Accessing Model objects from the locations
        logging.info(f" Accessing saved  model object  :{saved_model_path}")
        saved_model_object = load_pickle_object(file_path=saved_model_path)
        logging.info(f" Accessing artifact  model object  :{model_trained_artifact_path}")
        artifact_model_object = load_pickle_object(file_path=model_trained_artifact_path)
        
        # Comparing models based on mse score
        if saved_mse_score < artifact_model_mse:
            best_model_name = saved_model_name
            best_mse_score = saved_mse_score
            best_model_object = saved_model_object
        else:
            best_model_name = artifact_model_name
            best_mse_score = artifact_model_mse
            best_model_object = artifact_model_object
        
        return best_model_name, best_mse_score, best_model_object


    def __init__(self,
                    model_trainer_artifact:ModelTrainerTIMEArtifact):
        
        try:
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_evaluation_and_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.trained_model_object_file_path
            model_trained_report = self.model_trainer_artifact.model_report
            
            # Plot images 
            plot_artifact_path=self.model_trainer_artifact.prediction_image
                        
            # Saved Model file Paths
            saved_model_report_path = self.model_trainer_artifact.saved_report_file_path
            saved_model_path = self.model_trainer_artifact.saved_model_file_path

            
            
            # Loading the models - making saved Models directory 
            logging.info("Saved_models directory .....")
            os.makedirs(SAVED_MODEL_DIRECTORY,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(SAVED_MODEL_DIRECTORY):
                logging.info(f" Saved models Directory is enmpty ")
                logging.info(f" Artifact contents ------> Saved model Directory")
                # Saved files does not exits (Artifact -----> Saved Directory )
                                    # Making_model_report
                                    # Reading data from articaft 
                artifact_model_name,artifact_model_mse=self.read_model_info(file_path=model_trained_report)
                                    # Saving data to saved folder 
                self.save_model_report(model_name=artifact_model_name,mse=artifact_model_mse,model_report_path=saved_model_report_path)

                logging.info(f" Report Copied ")
            
                # Artifact model path 
                Artifact_model_path=model_trained_artifact_path
                logging.info(f" Artifact Model path : {Artifact_model_path}")
                model = load_pickle_object(file_path=Artifact_model_path)
                file_path=saved_model_path
                
                save_pickle_object(file_path=file_path, model=model)
                logging.info("Artifact Model saved.")
                
                # Plot image save
                destination_path=os.path.join(self.model_trainer_artifact.saved_model_file_path,'best_model_prediction')
                copy_image(source_path=plot_artifact_path,destination_path=destination_path)
                
            else:
                # Saved model data exists
                logging.info(" Contents found in Saved Directory ")
                logging.info(" Comapring Models ")
                best_model_name, best_mse_score, best_model_object=self.compare_models(saved_model_path=saved_model_path,
                                                                                    saved_model_report_path=saved_model_report_path,
                                                                                    model_trained_artifact_path=model_trained_artifact_path,
                                                                                    model_trained_report=model_trained_report)
                logging.info(f" Model choosen -----> {best_model_name}")
                
                logging.info(f" report saved")
                self.save_model_report(model_name=best_model_name,mse=best_mse_score,model_report_path=saved_model_report_path)
                
                logging.info(" Saving Model .....")
                
                
                file_path=saved_model_path
                save_pickle_object(file_path=file_path, model=best_model_object)
                logging.info(f"{best_model_name} Model saved.")
        
            
            
            
    
            logging.info("Model evaluated ans saved !")
            model_pusher_artifact = ModelPusherArtifact(message="Model Pushed succeessfully")
            return model_pusher_artifact
        except Exception as e:
            logging.error("Error occurred !")
            raise CustomException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")