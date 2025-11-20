import os
import sys
import numpy as np
import pickle
import sklearn

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    adjusted_rand_score
)
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def evaluate(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
            
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            logging.info("Predictions done successfully")
            
            train_mse, train_mae, train_rmse, train_score, train_adjusted_score = get_score(y_train, train_pred)
            test_mse, test_mae, test_rmse, test_score, test_adjusted_score = get_score(y_test, test_pred)
            
            report[list(models.keys())[i]] = test_mse
            report[list(models.keys())[i]] = test_mae
            report[list(models.keys())[i]] = test_rmse
            report[list(models.keys())[i]] = test_score
            report[list(models.keys())[i]] = test_adjusted_score
            
        return report
            
    
    except Exception as e:
        raise CustomException(e, sys)

def get_score(true, pred):
    try:
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        score = r2_score(true, pred)
        adjusted_score = adjusted_rand_score(true, pred)
        return mse, mae, rmse, score, adjusted_rand_score
    except Exception as e:
        raise CustomException(e, sys)
        