import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import tensorflow as tf
import mlflow.sklearn
import joblib

#matplotlib inline

# Set the experiment name to an experiment in the shared experiments folder
import mlflow
mlflow.set_experiment("/HCD")
import subprocess
from pyngrok import ngrok, conf
import getpass
from datetime import datetime

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow will create an experiment if it doesn't exist
experiment_name = "/HCD"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = '2djGHpy0PcKxJbwGrsud9ym07wO_fYDug4BGMdyu1kGweKvR'
port=5000
public_url = ngrok.connect(port).public_url
print(f' * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"')

# Load the dataset
url = 'heart_statlog_cleveland_hungary_final.csv'
df = pd.read_csv(url)

X = df.drop(['target'], axis = 1)
y = df.target.values


import pandas as pd
from sklearn.model_selection import train_test_split

# Load your original dataset
data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Split the dataset into train, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the splits into CSV files
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import warnings



# This code snippet builds and trains a neural network
#  using Keras, a high-level
# neural networks API running on top of TensorFlow.
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history= classifier.fit(X_train, y_train, batch_size = 10, epochs = 100,  validation_split=0.2)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy = history.history['accuracy']
validation_acc = history.history['val_accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']

# # Print evaluation metrics
# print(f'Accuracy: {accuracy}')
# print(f'Loss: {loss}')
# print(f'Validation accuracy: {validation_acc}')
# print(f'Validation loss: {validation_loss}')

mlflow.end_run()
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name = run_name) as mlflow_run:

    mlflow.set_experiment_tag("base_model", "classifier")
    mlflow.set_tag("optimizer", "keras.optimizers.Adam")
    mlflow.set_tag("loss", "sparse_categorical_crossentropy")


    # Log metrics
    mlflow.log_metrics({
        "train_loss": history.history["loss"][-1],
        "train_accuracy": history.history["accuracy"][-1],
        "val_loss": history.history["val_loss"][-1],
        "val_accuracy": history.history["val_accuracy"][-1]
    })


    
    
    # Save the model with MLflow
    mlflow.keras.log_model(classifier, "keras_model")

    

    
    


    # Log model to MLflow
    #mlflow.sklearn.log_model(model, "model")

    # Save the model using joblib
    joblib.dump(classifier, 'HCD_model.pkl')


    # # Save the model path for future reference
    model_path = "HCD_model.pkl"

    # # Log the model path to MLflow
    mlflow.log_artifact(model_path)

