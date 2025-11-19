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

def evaluate(true, pred):
    try:
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        score = r2_score(true, pred)
        adjusted_score = adjusted_rand_score(true, pred)
        return mse, mae, rmse, score, adjusted_score
    
    except Exception as e:
        raise CustomException(e, sys)