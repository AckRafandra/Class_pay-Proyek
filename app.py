import streamlit as st
import joblib
import numpy as np
import sklearn


# Load model
model = joblib.load('random_forest_model 2.pkl')

# Title and description
st.title("Prediksi Metode Pembayaran")
st.write("""
    Masukkan data pelanggan untuk memprediksi metode pembayaran yang paling sesuai.
    Kami akan membantu Anda memahami pola perilaku pelanggan berdasarkan input yang diberikan.
""")
# Add an image to make it more attractive (you can replace with your own image URL or file)
st.image("https://wallpapers.com/images/high/japanese-anime-aesthetic-19r6zi160sm63okj.webp", width=700)  # Replace with your image URL

# Input fields
st.subheader("Data Pelanggan")
age = st.number_input("Usia Pelanggan:", min_value=1, max_value=120, step=1)  # Tambahan input usia
gender = st.selectbox("Pilih Jenis Kelamin:", ["Laki-laki", "Perempuan"])

product_category = st.selectbox(
    "Pilih Kategori Produk:", 
    ["Clothing", "Electronics", "Fast Food", "Household Items", "Snacks", "Toys"]
)

city = st.selectbox(
    "Pilih Kota:", 
    ["Denpasar", "Jakarta", "Medan", "Semarang", "Surabaya", "Yogyakarta"]
)

quantity = st.number_input("Jumlah Barang:", min_value=1, step=1)
price_per_unit = st.number_input("Harga per Unit (dalam Rupiah):", min_value=500, max_value=100000000, step=500)

# Prediction button
if st.button("Prediksi Metode Pembayaran"):
    if 'model' in locals():
        # Preprocess input
        total_amount = quantity * price_per_unit

        features = np.array([
            age,  # Pastikan fitur usia ada
            quantity,
            price_per_unit,
            total_amount,
            1 if gender == "Laki-laki" else 0,
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

        try:
            # Predict
            prediction = model.predict(features)[0]
            payment_methods = {0: "E-wallet", 1: "Tunai", 2: "Kartu Kredit"}
            result = payment_methods[prediction]

            # Display result
            st.success(f"Prediksi Metode Pembayaran: {result}")

            # Display detailed price
            st.write(f"Total Harga: **Rp {total_amount:,.2f}**")
        except ValueError as e:
            st.error(f"Error prediksi: {e}")
    else:
        st.error("Model belum dimuat. Periksa file model.")
