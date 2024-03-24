from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your machine learning model
model = load_model('HCD_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    chest_pain_type = float(request.form['chest_pain_type'])
    resting_bp_s = float(request.form['resting_bp_s'])
    cholesterol = float(request.form['cholesterol'])
    fasting_blood_sugar = float(request.form['fasting_blood_sugar'])
    resting_ecg = float(request.form['resting_ecg'])
    max_heart_rate = float(request.form['max_heart_rate'])
    exercise_angina = float(request.form['exercise_angina'])
    oldpeak = float(request.form['oldpeak'])
    ST_slope = float(request.form['ST_slope'])

    # Create input array (including target)
    input_data = np.array([[age, sex, chest_pain_type, resting_bp_s, cholesterol, 
                            fasting_blood_sugar, resting_ecg, max_heart_rate,
                            exercise_angina, oldpeak, ST_slope]])

    # Make prediction
    prediction = model.predict(input_data)

    # Format prediction (assuming binary classification)
    if prediction > 0.5:
        result = 'Positive'
    else:
        result = 'Negative'

    # Pass the result back to the same HTML page
    return render_template('index.html', prediction_result=result)


if __name__ == '__main__':
    app.run(debug=True)
