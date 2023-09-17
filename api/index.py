# app.py

from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Specify the filename of your model
model_path = "../xgb_model.pkl"

# Check model loading
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print("Model loaded successfully.")

@app.route('/')
def mn():
    return "Jai Shree Ram!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request as JSON
        data = request.get_json()
        print(data)


        d = pd.DataFrame([data])
        print(d.dtypes)

        # Make predictions
        print(model.get_booster().feature_names)
        prediction = model.predict(d)
        print("Prediction:", prediction)

        # Convert the prediction to the desired format
        prediction = prediction.tolist()

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})
