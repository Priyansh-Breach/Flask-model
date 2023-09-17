# app.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained XGBoost model
with open('../xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return "Jai Shree Ram!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request as JSON
        data = request.get_json()
        # Perform inference using the loaded model
        prediction = model.predict(pd.DataFrame([data]))

        # Convert the prediction to the desired format
        # For example, if the prediction is a NumPy array, you can convert it to a list
        prediction = prediction.tolist()

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})
