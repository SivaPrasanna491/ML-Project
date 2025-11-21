import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/trainer.pkl'
            preprocessor = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomData(e, sys)
        
class CustomData():
    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation,
        reading_score,
        writing_score
        ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity,
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation = test_preparation
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_frame(self):
        try:
            return pd.DataFrame({
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": self.lunch,
            "test_preparation": [self.test_preparation],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        })
        except Exception as e:
            raise CustomException(e, sys)
        