import streamlit as st
import pandas as pd
import joblib

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

model_choice = st.selectbox("Pilih model machine learning:", list(MODEL_FILES.keys()))
pipeline = joblib.load(MODEL_FILES[model_choice])

tab1, tab2 = st.tabs(["Cek Komentar Manual", "Analisis CSV Batch"])

# ==== TAB 1: CEK KOMENTAR MANUAL ====
with tab1:
    user_input = st.text_area("Tulis komentar atau ulasan di sini:", height=120)
    if st.button("Cek Sentimen", key="cek_manual"):
        if user_input.strip() == "":
            st.warning("Tolong masukkan komentar terlebih dahulu!")
        else:
            pred = pipeline.predict([user_input])[0]
            if pred.lower() == "positive" or pred == 1:
                st.markdown(f'<p class="big-font positive">Sentimen: POSITIF 😊</p>', unsafe_allow_html=True)
            elif pred.lower() == "negative" or pred == 0:
                st.markdown(f'<p class="big-font negative">Sentimen: NEGATIF 😡</p>', unsafe_allow_html=True)
            else:
                st.info(f"Sentimen: {pred}")
        st.caption(f"Model aktif: {model_choice}")

# ==== TAB 2: UPLOAD CSV & BATCH ANALISIS ====
with tab2:
    uploaded_file = st.file_uploader("Upload file CSV untuk analisis sentimen massal", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Contoh data:", df.head())
        
        # Pilih kolom komentar
        comment_cols = [col for col in df.columns if 'review' in col.lower() or 'detail' in col.lower() or 'komentar' in col.lower() or 'comment' in col.lower()]
        col_selected = st.selectbox("Pilih kolom yang berisi komentar/ulasan:", df.columns, index=0 if len(comment_cols) == 0 else df.columns.get_loc(comment_cols[0]))
        
        # Tampilkan mapping nama kolom
        st.info(f"Nama kolom akan dimapping ke 'Komentar'. Kolom dipilih: {col_selected}")

        if st.button("Analisis Sentimen CSV"):
            komentar_list = df[col_selected].astype(str).fillna("-").tolist()
            prediksi = pipeline.predict(komentar_list)
            df['Komentar'] = df[col_selected]
            df['Sentimen_Prediksi'] = prediksi
            st.success("Analisis selesai! Berikut beberapa hasil:")
            st.dataframe(df[['Komentar', 'Sentimen_Prediksi']].head(20))

            # Download hasil prediksi
            csv_hasil = df.to_csv(index=False)
            st.download_button("Download hasil prediksi (CSV)", csv_hasil, file_name="hasil_prediksi_sentimen.csv", mime="text/csv")

st.markdown("""
---
App by Muhammad Irfan Karim, powered by Streamlit.
""")
