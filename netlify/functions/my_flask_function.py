from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Ambil data JSON dari frontend
    features = np.array([[  # Transform data jadi input model
        data['age'],
        data['quantity'],
        data['price_per_unit'],
        data['quantity'] * data['price_per_unit'],
        1 if data['gender'] == 'Male' else 0,
        1 if data['product_category'] == 'Clothing' else 0,
        1 if data['product_category'] == 'Electronics' else 0,
        1 if data['city'] == 'Denpasar' else 0,
        1 if data['city'] == 'Jakarta' else 0,
        1 if data['city'] == 'Surabaya' else 0
    ]])

    prediction = model.predict(features)[0]
    payment_methods = {0: 'Cash', 1: 'E-wallet', 2: 'Credit Card'}
    return jsonify({'prediction_text': f'Metode Pembayaran: {payment_methods[prediction]}'})

# Convert Flask app ke Netlify Function handler
from flask_lambda import FlaskLambda
app = FlaskLambda(app)
