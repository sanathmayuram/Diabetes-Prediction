from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

grid_model = pickle.load(open('grid.pkl', 'rb'))
standed_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # Retrieve form data for Diabetes Prediction
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

# Optional: create a list or numpy array to pass to your model
        input_features = [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]]

        new_data_scaled = standed_scaler.transform(input_features)

        prediction = grid_model.predict(new_data_scaled)[0]  # extract the scalar 0 or 1

# Now assign readable labels
        if prediction == 1:
            result = "Positive"  # Diabetes present
        else:
            result = "Negative"  # No diabetes

        return render_template("home.html",results=result)
    else:
        return render_template('home.html')
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)