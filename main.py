import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model
with open("random_forest_model.pkl", "rb") as model_file:
    pipe = pickle.load(model_file)

# Load training data
training_data = pd.read_csv("Cleaned_data.csv")

# Extract features (X) and target (y)
X_train = training_data.drop(columns=["price", "price_unit", "money in Cr", "locality", "region"])
y_train = training_data["money in Cr"]

# Fit the model with training data
pipe.fit(X_train, y_train)

# Define the feature names used during training
feature_names = X_train.columns.tolist()

@app.route('/')
def index():
    locations = sorted(training_data['region'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Generate a random value between 0 and 167 for the location
    location = int(request.form.get('location'))
    bhk = int(request.form.get('bhk'))
    house_type = request.form.get('type')
    area = float(request.form.get('area'))

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'area': [area],
        'Unnamed: 0': [location],
        'bhk': [bhk],
        'type': [house_type]
    })

    # Reorder columns to match the order used during training
    input_data = input_data[feature_names]

    # Make prediction
    prediction = pipe.predict(input_data)[0] * 1e7
    

    # Pass the prediction back to the HTML template
    return render_template('index.html', locations=sorted(training_data['region'].unique()), prediction=np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
