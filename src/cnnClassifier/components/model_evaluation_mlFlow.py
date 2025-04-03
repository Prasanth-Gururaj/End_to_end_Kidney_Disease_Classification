import tensorflow as tf
from pathlib import Path
import os  
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        try:
            import dagshub
            dagshub.init(repo_owner='Prasanth-Gururaj', repo_name='End_to_end_Kidney_Disease_Classification', mlflow=True)
        except ImportError:
            print("DAGsHub not installed. Continuing with direct MLflow configuration.")
        
        with mlflow.start_run():
            try:
                # Log the model
                mlflow.keras.log_model(self.model, "model")
                
                # Log metrics from evaluation
                mlflow.log_metrics({
                    "loss": self.score[0], 
                    "accuracy": self.score[1]
                })
                
                # Log all parameters from config
                # First, convert any non-string parameters to strings
                params_dict = {}
                
                # Add specific parameters you always want to track
                params_dict["model_type"] = "VGG16"
                params_dict["batch_size"] = self.config.params_batch_size
                params_dict["image_size"] = str(self.config.params_image_size)
                
                # Add all parameters from the config if they exist
                if hasattr(self.config, 'all_params'):
                    # Flatten nested dictionaries if needed
                    for key, value in self.config.all_params.items():
                        # Handle nested dictionaries
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                params_dict[f"{key}_{nested_key}"] = str(nested_value)
                        else:
                            params_dict[key] = str(value)
                
                # Log all parameters
                mlflow.log_params(params_dict)
                
                print("Model, metrics, and parameters successfully logged to MLflow")
                    
            except ModuleNotFoundError as e:
                if "distutils._modified" in str(e):
                    print("Encountered setuptools/distutils conflict. Trying alternative logging method...")
                    # Save the model locally first
                    local_model_path = "temp_model_artifacts"
                    os.makedirs(local_model_path, exist_ok=True)
                    self.model.save(local_model_path)
                    
                    # Still log metrics and parameters
                    mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
                    
                    # Log the same parameters as above
                    params_dict = {
                        "model_type": "VGG16", 
                        "batch_size": self.config.params_batch_size,
                        "image_size": str(self.config.params_image_size)
                    }
                    
                    # Add all other parameters if available
                    if hasattr(self.config, 'all_params'):
                        for key, value in self.config.all_params.items():
                            if isinstance(value, dict):
                                for nested_key, nested_value in value.items():
                                    params_dict[f"{key}_{nested_key}"] = str(nested_value)
                            else:
                                params_dict[key] = str(value)
                    
                    mlflow.log_params(params_dict)
                    print("Logged model metrics and parameters instead")
                else:
                    raise
            except Exception as e:
                print(f"Error in logging to MLflow: {e}")
                raise