from src.datascience import logger


from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
STAGE_NAME = " Data Ingestion Stage"

try:
    logger.info(f'>>>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f'>>>>>>>> stage {STAGE_NAME} completed <<<<<<<')

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f'>>>>>> stage {STAGE_NAME} started <<<<<<')
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    data_validation.initiate_data_validation()
    logger.info(f'>>>>>> stage {STAGE_NAME} completed <<<<<<')  

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f'>>>>>> stage {STAGE_NAME} started <<<<<<')
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f'>>>>>> stage {STAGE_NAME} completed <<<<<<')


except Exception as e:
    logger.exception(e)
    raise e
    
             
STAGE_NAME = 'Model Training Stage'

try:
    logger.info(f'>>>>>> stage {STAGE_NAME} started <<<<<<')
    model_training = ModelTrainerPipeline()
    model_training.initiate_model_training()
    logger.info(f'>>>>>> stage {STAGE_NAME} completed <<<<<<')
    
except Exception as e:
    logger.exception(e)
    raise e   


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f'>>>>>> stage {STAGE_NAME} started <<<<<<')
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f'>>>>>> stage {STAGE_NAME} completed <<<<<<')

except Exception as e:
    logger.exception(e)
    raise e

# These are the stages of the training pipeline. 

# This is now the prediction pipeline.
