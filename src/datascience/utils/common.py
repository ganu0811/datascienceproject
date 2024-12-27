import os
import sys
import yaml
from src.datascience import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

@ensure_annotations

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns
    

    Args:
        path_to_yaml (str):path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox type
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)  # Reading a yaml file
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    
    except BoxValueError:
        raise ValueError('yaml file is empty') # If there is any error related to Value error then it will be handled by the BoxValueError
    except Exception as e: 
        raise e
    

@ensure_annotations

def create_directories(path_to_directories: list, verbose= True):
    """create list of directories

    Args:
        path_to_directories (list): List of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to 
    """
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"directory created at {path}")
            

@ensure_annotations

def save_json(path:Path, data: dict):
    """save json data
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        
    
    logger.info(f"json file saved at: {path}")    


@ensure_annotations

def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox:data as class attributes instead of dict
    """
    
    with open(path) as f:
        content = json.load(f)
    
    
    logger.info(f'json file loaded successfully from: {path}')
    return ConfigBox(content)

@ensure_annotations

def save_bin(data: Any, path: Path):  # Save the model
    """save binary file
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")
    

@ensure_annotations

def load_bin(path: Path) -> Any:    # Load the model
    """load binary data
    
    Args:
        path (Path): path to binary file
    
    Returns:
        Any: object stored in the file
    """
    
    data = joblib.load(path)
    logger.info(f'binary file loaded from: {path}')
    return data    