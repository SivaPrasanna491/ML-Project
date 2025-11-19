import os
import sys
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_tranformer_object(self):
        try:
            numerical_features = ["writing_score", "reading_score"]
            cat_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical pipeline created succesfully")
            logging.info("Categorical pipeline created succesfully")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )
            return preprocessor
        except Exception as  e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading training and testing datasets completed successfully")
            
            preprocessing_obj = self.get_data_tranformer_object()
            
            logging.info("Preprocessing obj loaded successfully")
            
            X_train_df = train_df.drop("math_score", axis=1)
            y_train_df = train_df['math_score']
            
            X_test_df = test_df.drop("math_score", axis=1)
            y_test_df = test_df['math_score']
            
            logging.info("Independent and dependent features loaded successfully")
            
            X_train_arr = preprocessing_obj.fit_transform(X_train_df)
            X_test_arr = preprocessing_obj.transform(X_test_df)
            
            logging.info("Handling null and categorical features completed successfully")
            
            train_arr = np.c_[X_train_arr, y_train_df]
            test_arr = np.c_[X_test_arr, y_test_df]
            
            save_object(
                file_path= self.transformation_config.preprocessed_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Saved preprocessing data")
            
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessed_obj_file_path
            )        
        except Exception as e:
            raise CustomException(e, sys)
            
        
        

