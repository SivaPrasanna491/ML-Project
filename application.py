from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST`'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation = request.form.get("test_preparation_course"),
            reading_score = request.form.get("reading_score"),
            writing_score = request.form.get("writing_score")
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)
        
        pred = PredictPipeline()
        
        preds = pred.predict(pred_df)
        
        # print(preds)
        return render_template('home.html', preds=preds[0])

if __name__=="main":
    app.run(host="0.0.0.0")
        
