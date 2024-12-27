
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories # This is for reading the yaml file in self.config below
from src.datascience.entity.config_entity import (DataIngestionConfig, DataValidationConfig) # This is the input to the data ingestion pipeline.

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root]) # This is to create the artifacts directory
        
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        
        )
        
        # This is the use of ConfigBox as we can directly fetch the value using 'config.'
        return data_ingestion_config
    
    def get_data_validation_config(self):
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_data_dir=config.unzip_data_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema)
        
        return data_validation_config