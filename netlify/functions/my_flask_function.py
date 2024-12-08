from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    gender = data['gender']
    age = int(data['age'])
    product_category = data['product_category']
    city = data['city']
    quantity = int(data['quantity'])
    price_per_unit = float(data['price_per_unit'])

    # Prediksi seperti biasa
    features = np.array([[  # Masukkan logika Anda di sini
        age,
        quantity,
        price_per_unit,
        quantity * price_per_unit,
        1 if gender == 'Male' else 0,
        1 if product_category == 'Clothing' else 0,
        1 if product_category == 'Electronics' else 0,
        1 if city == 'Denpasar' else 0,
        1 if city == 'Jakarta' else 0,
        1 if city == 'Makassar' else 0,
        1 if city == 'Medan' else 0,
        1 if city == 'Semarang' else 0,
        1 if city == 'Surabaya' else 0,
        1 if city == 'Yogyakarta' else 0
    ]])

    prediction = model.predict(features)[0]
    payment_methods = {0: 'Cash', 1: 'E-wallet', 2: 'Credit Card'}
    result = payment_methods[prediction]

    return jsonify({"prediction_text": f"Metode Pembayaran: {result}"})


# Convert Flask app ke Netlify Function handler
from flask_lambda import FlaskLambda
app = FlaskLambda(app)
