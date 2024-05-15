from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# It is giving us the entry point of the application
application = Flask(__name__)
# Assign the application to a variable to create the route
app = application

# Creation of the route for a home page

@app.route('/')

def index():
    # we have the route(/) decorator applied to the index() function, 
    #meaning that the function will handle requests to the root URL ’/’
    return render_template('index.html') 

# REST API
@app.route('/predictdata', methods=['GET','POST'])

def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")
    
    else:
        # Get the datapoints
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('writing_score')),
            writing_score = float(request.form.get('reading_score'))

        )

        # Get the dataframe to predict
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        # Load the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        # Perform the prediction
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html', results = results[0])
    

if __name__ == "__main__":
    # Map the app to 127.0.0.1 with default port being 5000
    # To access the site use http://127.0.0.1:5000/
    app.run(host="0.0.0.0")        