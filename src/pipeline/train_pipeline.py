from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import Model_Trainer, Model_trainer_config

if __name__=="main":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, test_arr, file_path = transformation.initiate_data_transformation(train_path, test_path)
    
    model_trainer = Model_Trainer()
    score = model_trainer.initiate_model_training(train_arr, test_arr, file_path)
    
    print(score)