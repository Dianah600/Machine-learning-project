#Include the requests library.
import requests

# Replace with the actual URL where your Flask app is running
url = 'http://127.0.0.1:5000/predict'

# Sample JSON data

data = {
    'age': 50,
    'sex': 0,
    'chest pain type': 2,
    'resting bp s': 130, 
    'cholesterol': 200,
    'fasting blood sugar': 1, 
    'resting ecg': 0, 
    'max heart rate':120,
    'exercise angina':0, 
    'oldpeak':1.0, 
    'ST slope':2, 
    'target':0
}



# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
