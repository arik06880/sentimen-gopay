# Dashboard Analisis Sentimen GoPay

Dashboard interaktif untuk analisis sentimen ulasan GoPay di Google Play Store.

## Fitur
- Prediksi sentimen real-time dari teks input
- Visualisasi distribusi dan tren sentimen
- Hasil pemodelan topik LDA per kelas sentimen

## Cara Deploy ke Streamlit Community Cloud

### 1. Siapkan repository GitHub
Pastikan struktur folder seperti ini:
```
repo/
├── app.py
├── requirements.txt
├── model_nb.pkl
├── tfidf_vectorizer.pkl
├── gopay_preprocessed.csv
├── hasil_topik_lda.csv
└── coherence_scores.csv
```

### 2. Upload file dari hasil notebook
File yang dibutuhkan (dari hasil notebook Colab):
- `model_nb.pkl` — dari notebook 03
- `tfidf_vectorizer.pkl` — dari notebook 03
- `gopay_preprocessed.csv` — dari notebook 02
- `hasil_topik_lda.csv` — dari notebook 04
- `coherence_scores.csv` — dari notebook 04

### 3. Deploy
1. Buka https://share.streamlit.io
2. Login dengan akun GitHub
3. Klik **New app**
4. Pilih repository, branch, dan file `app.py`
5. Klik **Deploy**

## Periode Data
Januari – Maret 2026

## Metode
- Klasifikasi: Multinomial Naive Bayes
- Pemodelan topik: Latent Dirichlet Allocation (LDA)
- Penentuan jumlah topik: Coherence score (c_v)
