import os
import sys
import numpy as np
import pandas as pd
import sklearn

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import(
    root_mean_squared_error,
    r2_score,
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor

from src.utils import *

@dataclass
class Model_trainer_config:
    trained_obj_file_path = os.path.join('artifacts', 'trained.pkl')

class Model_Trainer:
    def __init__(self):
        self.trainer_config = Model_trainer_config()
    
    def initiate_model_training(self, train_data, test_data, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_data[:, :-1],
                train_data[:,-1],
                test_data[:, :-1],
                test_data[:,-1],
            )
            
            logging.info("Loading the models")
            
            models = {
                "Linear Regressor": LinearRegression(),
                "Ridge Regressor": Ridge(),
                "Lasso Regressor": Lasso(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
            }
            
            logging.info("Training the model")
            
            report = {}
            
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                logging.info("Predictions done successfully")
                
                train_mse, train_mae, train_rmse, train_score, train_adjusted_score = evaluate(y_train, train_pred)
                test_mse, test_mae, test_rmse, test_score, test_adjusted_score = evaluate(y_test, test_pred)
                
                report[list(models.keys())[i]] = test_mse
                report[list(models.keys())[i]] = test_mae
                report[list(models.keys())[i]] = test_rmse
                report[list(models.keys())[i]] = test_score
                report[list(models.keys())[i]] = test_adjusted_score
            
            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            
            logging.info("Best model found")
            
            save_object(
                file_path=self.trainer_config.trained_obj_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            return r2_score(predicted, y_test)
        except Exception as e:
            raise CustomException(e, sys)

