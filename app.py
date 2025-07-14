import streamlit as st
import joblib

# Pilihan model yang tersedia
MODEL_FILES = {
    "Logistic Regression": "model_logreg_sentimen.joblib",
    "SVM": "model_svm_sentimen.joblib",
    "Naive Bayes": "model_nb_sentimen.joblib",
    "Random Forest": "model_rf_sentimen.joblib"
}

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖", layout="centered")
st.title("Sentiment Analyzer App 🤖")
st.caption("Cek sentimen komentar/kata-kata secara otomatis menggunakan model Machine Learning Indonesia")

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    font-weight: bold;
}
.positive { color: #1fab89; font-weight: bold; }
.negative { color: #ff4d4d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Pilihan model
model_choice = st.selectbox("Pilih model machine learning:", list(MODEL_FILES.keys()))

# Input text
user_input = st.text_area("Tulis komentar atau ulasan di sini:", height=120)

# Tombol prediksi
if st.button("Cek Sentimen"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan komentar terlebih dahulu!")
    else:
        # Load model sesuai pilihan user
        pipeline = joblib.load(MODEL_FILES[model_choice])
        # Predict
        pred = pipeline.predict([user_input])[0]
        # Warna hasil
        if pred.lower() == "positive" or pred == 1:
            st.markdown(f'<p class="big-font positive">Sentimen: POSITIF 😊</p>', unsafe_allow_html=True)
        elif pred.lower() == "negative" or pred == 0:
            st.markdown(f'<p class="big-font negative">Sentimen: NEGATIF 😡</p>', unsafe_allow_html=True)
        else:
            st.info(f"Sentimen: {pred}")

    # (Opsional) Tampilkan model yang dipakai
    st.caption(f"Model aktif: {model_choice}")

st.markdown("""
---
App by Muhammad Irfan Karim, powered by Streamlit.
""")
