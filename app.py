import streamlit as st
import joblib
import numpy as np

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

# Input fields with more detailed descriptions and styling
st.subheader("Data Pelanggan")
gender = st.selectbox("Pilih Jenis Kelamin:", ["Laki-laki", "Perempuan"])

# Batasan usia minimal 1 tahun dan maksimal 120 tahun
age = st.number_input("Usia (tahun):", min_value=1, max_value=120, step=1, format="%d")

# Update product categories
product_category = st.selectbox(
    "Pilih Kategori Produk:", ["Pakaian", "Elektronik", "Kecantikan", "Perlengkapan Rumah Tangga", "Snack", "Makanan Cepat Saji", "Mainan"]
)

city = st.selectbox(
    "Pilih Kota:", 
    ["Jakarta", "Bandung", "Surabaya", "Medan", "Yogyakarta", "Denpasar", "Semarang"]
)

# Batasan harga minimal 500 dan maksimal 100 juta
quantity = st.number_input("Jumlah Barang:", min_value=1, step=1)

# Mengubah batasan harga per unit menjadi minimal 500 dan maksimal 100 juta
price_per_unit = st.number_input("Harga per Unit (dalam Rupiah):", min_value=500, max_value=100000000, step=500)

# Prediction button with a custom style
if st.button("Prediksi Metode Pembayaran"):
    # Preprocess input
    features = np.array([[ 
        age,
        quantity,
        price_per_unit,
        quantity * price_per_unit,
        1 if gender == "Laki-laki" else 0,
        1 if product_category == "Pakaian" else 0,
        1 if product_category == "Elektronik" else 0,
        1 if product_category == "Kecantikan" else 0,
        1 if product_category == "Perlengkapan Rumah Tangga" else 0,
        1 if product_category == "Snack" else 0,
        1 if product_category == "Makanan Cepat Saji" else 0,
        1 if product_category == "Mainan" else 0,
        1 if city == "Denpasar" else 0,
        1 if city == "Jakarta" else 0,
        1 if city == "Makassar" else 0,
        1 if city == "Medan" else 0,
        1 if city == "Semarang" else 0,
        1 if city == "Surabaya" else 0,
        1 if city == "Yogyakarta" else 0,
    ]])

    # Predict
    prediction = model.predict(features)[0]
    payment_methods = {0: "Tunai", 1: "E-wallet", 2: "Kartu Kredit"}
    result = payment_methods[prediction]

    # Display result with a custom message
    st.success(f"Prediksi Metode Pembayaran: {result}")

    # Display detailed price in Rupiah format
    total_price = quantity * price_per_unit
    st.write(f"Total Harga: **Rp {total_price:,.2f}**")
