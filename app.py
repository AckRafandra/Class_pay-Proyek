from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model
model = joblib.load(r'class_pay proyek\random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari form
    gender = request.form['gender']
    age = int(request.form['age'])
    product_category = request.form['product_category']
    city = request.form['city']
    quantity = int(request.form['quantity'])
    price_per_unit = float(request.form['price_per_unit'])

    # Mengubah input ke format model
    features = np.array([[
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

    # Prediksi
    prediction = model.predict(features)[0]
    payment_methods = {0: 'Cash', 1: 'E-wallet', 2: 'Credit Card'}
    result = payment_methods[prediction]

    return render_template('index.html', prediction_text=f'Metode Pembayaran: {result}')

if __name__ == "__main__":
    app.run(debug=True)
