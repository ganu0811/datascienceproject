import os
from src.datascience import logger
from src.datascience.entity.config_entity import (DataTransformationConfig)
import pandas as pd
from sklearn.model_selection import train_test_split


class DataTransformation:
    
    def __init__(self, config:DataTransformationConfig):
        
        self.config = config

        # Note: Different data transformation techniques such as Scaler, PCA and others can be added here
        # All kinds of EDA in ML Cycle are performed here before passing this data to the model
        
    
    def train_test_splitting(self):
        
        data = pd.read_csv(self.config.data_path)
        
        # Splitting the data into train and test
        
        train, test = train_test_split(data)
        
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index = False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index = False)
        
        
        logger.info('Splitted data into training and testing')
        logger.info(train.shape)
        logger.info(test.shape)
        
        print(train.shape)
        print(test.shape)
        
        
        
        