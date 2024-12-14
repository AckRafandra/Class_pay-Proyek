import streamlit as st
import joblib
import numpy as np
import sklearn

# Load model
model = joblib.load('random_forest_model_last2.pkl')

# Title and description
st.title("Prediksi Metode Pembayaran")
st.write("""
    Masukkan data pelanggan untuk memprediksi metode pembayaran yang paling sesuai.
    Kami akan membantu Anda memahami pola perilaku pelanggan berdasarkan input yang diberikan.
""")

# Add an image to make it more attractive
st.image("https://wallpapers.com/images/high/japanese-anime-aesthetic-19r6zi160sm63okj.webp", width=700)

# Input fields
st.subheader("Data Pelanggan")
age = st.number_input("Usia Pelanggan:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Pilih Jenis Kelamin:", ["Laki-laki", "Perempuan"])

# Select product category
product_category = st.selectbox(
    "Pilih Kategori Produk:", 
    ["Books and Stationery", "Clothing", "Electronics", "Food and Beverages", "Health and Beauty", "Home and Living", "Toys and Games"]
)

# Select city
city = st.selectbox(
    "Pilih Kota:", 
    ["Surabaya", "Bandung", "Jakarta", "Yogyakarta", "Medan", "Semarang", "Denpasar"]
)

quantity = st.number_input("Jumlah Barang:", min_value=1, step=1)
price_per_unit = st.number_input("Harga per Unit (dalam Rupiah):", min_value=500, max_value=100000000, step=500)

# Prediction button
if st.button("Prediksi Metode Pembayaran"):
    if 'model' in locals():
        # Preprocess input
        total_amount = quantity * price_per_unit

        # One-hot encode input
        features = np.array([
            age,
            quantity,
            price_per_unit,
            total_amount,
            1 if gender == "Laki-laki" else 0,
            1 if product_category == "Books and Stationery" else 0,
            1 if product_category == "Clothing" else 0,
            1 if product_category == "Electronics" else 0,
            1 if product_category == "Food and Beverages" else 0,
            1 if product_category == "Health and Beauty" else 0,
            1 if product_category == "Home and Living" else 0,
            1 if product_category == "Toys and Games" else 0,
            1 if city == "Surabaya" else 0,
            1 if city == "Bandung" else 0,
            1 if city == "Jakarta" else 0,
            1 if city == "Yogyakarta" else 0,
            1 if city == "Medan" else 0,
            1 if city == "Semarang" else 0,
            1 if city == "Denpasar" else 0,
        ]).reshape(1, -1)

        try:
            # Predict
            prediction = model.predict(features)[0]
            payment_methods = {0: "E-Wallet", 1: "Credit Card", 2: "Cash", 3: "Bank Transfer"}
            result = payment_methods[prediction]

            # Display result
            st.success(f"Prediksi Metode Pembayaran: {result}")

            # Display detailed price
            st.write(f"Total Harga: **Rp {total_amount:,.2f}**")
        except ValueError as e:
            st.error(f"Error prediksi: {e}")
    else:
        st.error("Model belum dimuat. Periksa file model.")
