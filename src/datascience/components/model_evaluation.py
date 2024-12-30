import os

from src.datascience import logger

from src.datascience.entity.config_entity import (ModelEvaluationConfig
                                                )
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
from urllib.parse import urlparse
from pathlib import Path
from src.datascience.utils.common import read_yaml, create_directories, save_json

import os

# os.environ['MLFLOW_TRACKING_URI']= 'https://dagshub.com/ganu0811/datascienceproject.mlflow'
# os.environ['MLFLOW_TRACKING_USERNAME']= 'ganu0811'
# os.environ['MLFLOW_TRACKING_PASSWORD']= '408b0d3ca30244534b813dfd774bda4b64cc1013'


class ModelEvaluation:
    
    def __init__(self, config=ModelEvaluationConfig):
        
        self.config = config
    
    def eval_metrics(self, actual, pred):
        
        rmse = np.sqrt(mean_absolute_error(actual, pred))
        mse = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        return rmse, mse, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        
        with mlflow.start_run():
            
            predicted_qualities = model.predict(test_x)
            
            (rmse, mse, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving the log/metric parameters
            metrics_file = Path(self.config.metric_file_name)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.touch(exist_ok=True)
            scores = {'rmse': rmse, 'mse': mse, 'r2': r2}
            save_json(path = Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric('rmse',rmse)
            mlflow.log_metric('mse',mse)
            mlflow.log_metric('r2',r2)
            
            # Model registry does not work with file store
            
            if tracking_url_type_store != 'file':
                
                # Register the mode
                # There are other ways to use the model registry, which depends on the use case,
                
                mlflow.sklearn.log_model(model, "model", registered_model_name= "ElasticNetModel")
            
            else:
                mlflow.sklearn.log_model(model, 'model')
                 
            
            
        
        