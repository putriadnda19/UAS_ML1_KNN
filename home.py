import streamlit as st
from PIL import Image

def app():
    st.markdown(
        """
        <style>
        .css-2trqyj {
            font-family: 'Times New Roman', Times, serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Aplikasi Prediksi Pendapatan Orang Dewasa")
    st.write("Selamat Datang! Aplikasi prediksi pendapatan orang dewasa membantu pengguna merencanakan keuangan, membuat anggaran, dan mengambil keputusan karier berdasarkan estimasi penghasilan")
    # Load gambar dengan latar belakang terhapus
    image_path = 'output1.png'  # Ganti dengan path gambar Anda
    image = Image.open(image_path)

    # Tampilkan gambar dengan latar belakang terhapus
    st.image(image, use_column_width=800)

