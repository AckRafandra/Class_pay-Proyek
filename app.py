import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('random_forest_model_2.pkl')

# Title and description
st.title("Prediksi Metode Pembayaran")
st.write("""
    Masukkan data pelanggan untuk memprediksi metode pembayaran yang paling sesuai.
""")

# Input fields
st.subheader("Data Pelanggan")
age = st.number_input("Usia (tahun):", min_value=1, max_value=120, step=1, format="%d")
quantity = st.number_input("Jumlah Barang:", min_value=1, step=1)
price_per_unit = st.number_input("Harga per Unit (dalam Rupiah):", min_value=500, max_value=100000000, step=500)
gender = st.selectbox("Pilih Jenis Kelamin:", ["Laki-laki", "Perempuan"])

product_category = st.selectbox(
    "Pilih Kategori Produk:", 
    ["Clothing", "Electronics", "Fast Food", "Household Items", "Snacks", "Toys"]
)

city = st.selectbox(
    "Pilih Kota:", 
    ["Denpasar", "Jakarta", "Medan", "Semarang", "Surabaya", "Yogyakarta"]
)

# Prediction button
if st.button("Prediksi Metode Pembayaran"):
    # Preprocess input
    total_amount = quantity * price_per_unit

    # Encoding input sesuai dengan fitur pelatihan
    features = np.array([
        age,
        quantity,
        price_per_unit,
        total_amount,
        1 if gender == "Laki-laki" else 0,  # Gender_Male
        1 if product_category == "Clothing" else 0,
        1 if product_category == "Electronics" else 0,
        1 if product_category == "Fast Food" else 0,
        1 if product_category == "Household Items" else 0,
        1 if product_category == "Snacks" else 0,
        1 if product_category == "Toys" else 0,
        1 if city == "Denpasar" else 0,
        1 if city == "Jakarta" else 0,
        1 if city == "Medan" else 0,
        1 if city == "Semarang" else 0,
        1 if city == "Surabaya" else 0,
        1 if city == "Yogyakarta" else 0,
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    payment_methods = {0: "Tunai", 1: "E-wallet", 2: "Kartu Kredit"}
    result = payment_methods[prediction]

    # Display result
    st.success(f"Prediksi Metode Pembayaran: {result}")

    # Display detailed price
    st.write(f"Total Harga: **Rp {total_amount:,.2f}**")
