# ============================================================
# app.py — Dashboard Analisis Sentimen GoPay
# Streamlit Community Cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import re
from collections import Counter

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title='Analisis Sentimen GoPay',
    page_icon='💚',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Konstanta ───────────────────────────────────────────────
WARNA       = {'positif': '#2196F3', 'negatif': '#F44336', 'netral': '#FF9800'}
LABEL_ORDER = ['positif', 'negatif', 'netral']
EMOJI       = {'positif': '😊 Positif', 'negatif': '😠 Negatif', 'netral': '😐 Netral'}

# ── Kamus slang (sama dengan notebook 02) ───────────────────
KAMUS_SLANG = {
    'gk':'tidak','ga':'tidak','gak':'tidak','ngga':'tidak',
    'nggak':'tidak','tdk':'tidak','kaga':'tidak','kagak':'tidak',
    'ko':'kok','bgt':'sangat','banget':'sangat','bngt':'sangat',
    'sy':'saya','aq':'saya','gw':'saya','gue':'saya','gua':'saya',
    'km':'kamu','lo':'kamu','lu':'kamu',
    'yg':'yang','dg':'dengan','dgn':'dengan','dr':'dari',
    'utk':'untuk','tuk':'untuk','buat':'untuk','krn':'karena',
    'karna':'karena','tp':'tapi','tpi':'tapi','pdhl':'padahal',
    'sm':'sama','udh':'sudah','udah':'sudah','sdh':'sudah',
    'dah':'sudah','blm':'belum','belom':'belum','blom':'belum',
    'msih':'masih','msh':'masih','lg':'lagi','lgi':'lagi',
    'dlu':'dulu','dl':'dulu','mnt':'menit','dtk':'detik',
    'smpe':'sampai','sampe':'sampai','smpai':'sampai',
    'jg':'juga','bs':'bisa','bsa':'bisa','jt':'juta','rb':'ribu',
    'aja':'saja','aj':'saja','emng':'memang','emang':'memang',
    'ad':'ada','gimana':'bagaimana','gmn':'bagaimana',
    'gopey':'gopay','bayr':'bayar','nasabh':'nasabah',
    'pembayran':'pembayaran','bgus':'bagus','bgs':'bagus',
    'peroses':'proses','sya':'saya','sanggat':'sangat',
    'trimakasih':'terima kasih','kalo':'kalau','nyoba':'mencoba',
    'app':'aplikasi','apk':'aplikasi','tf':'transfer',
    'topup':'isi ulang','mantap':'bagus','mantul':'bagus',
    'keren':'bagus','oke':'ok','okey':'ok','okay':'ok',
    'jelek':'buruk','parah':'buruk','ancur':'buruk',
    'lmbt':'lambat','lemot':'lambat','ribet':'rumit','susah':'sulit',
    'eror':'error','err':'error','loading':'memuat','load':'memuat',
    'update':'perbarui','updt':'perbarui','duit':'uang','duwit':'uang',
    'thanks':'terima kasih','thx':'terima kasih','makasih':'terima kasih',
    'mksh':'terima kasih','tq':'terima kasih',
    'tlg':'tolong','tlong':'tolong',
}

STOPWORDS_TAMBAHAN = {
    'sih','deh','loh','lah','dong','kan','nih','tuh','nya','ku','mu',
    'kayak','kaya','yah','wah','hah','eh','ah','oh','iya','ya',
    'hai','hay','hei','gopay','gojek','google','play','store',
    'android','hp','ok','oke',
}

# ── Load model & data ───────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_nb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

@st.cache_data
def load_data():
    df = pd.read_csv('gopay_preprocessed.csv', encoding='utf-8-sig')
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['bulan']   = df['tanggal'].dt.to_period('M').astype(str)
    return df

@st.cache_data
def load_topik():
    return pd.read_csv('hasil_topik_lda.csv', encoding='utf-8-sig')

# ── Fungsi preprocessing (sama dengan notebook 02) ──────────
def preprocessing_prediksi(teks):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer          = StemmerFactory().create_stemmer()
    stopwords_default = set(StopWordRemoverFactory().get_stop_words())
    all_stopwords    = stopwords_default.union(STOPWORDS_TAMBAHAN)

    teks = teks.lower()
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'[@#]\w+', ' ', teks)
    teks = re.sub(r'\d+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    teks = ' '.join([KAMUS_SLANG.get(k, k) for k in teks.split()])
    kata = [k for k in teks.split() if k not in all_stopwords]
    kata = [stemmer.stem(k) for k in kata]
    return ' '.join(kata)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/GoPay_logo.svg/320px-GoPay_logo.svg.png', width=160)
    st.markdown('## Navigasi')
    halaman = st.radio(
        'Pilih halaman:',
        ['🔍 Prediksi Sentimen', '📊 Hasil Analisis', '📋 Topik LDA'],
        label_visibility='collapsed'
    )
    st.markdown('---')
    st.markdown(
        '**Tentang Dashboard**\n\n'
        'Dashboard ini merupakan hasil penelitian skripsi '
        'analisis sentimen ulasan GoPay di Google Play Store '
        'menggunakan Naive Bayes dan LDA.\n\n'
        '📅 Periode data: Jan–Mar 2026'
    )

# ── Load semua data ──────────────────────────────────────────
try:
    model, tfidf = load_model()
    df           = load_data()
    df_topik     = load_topik()
except Exception as e:
    st.error(f'Gagal memuat file: {e}')
    st.stop()

# ============================================================
# HALAMAN 1 — PREDIKSI SENTIMEN
# ============================================================
if halaman == '🔍 Prediksi Sentimen':
    st.title('🔍 Prediksi Sentimen')
    st.markdown(
        'Masukkan teks ulasan GoPay di bawah ini untuk mengetahui '
        'sentimen yang terkandung.'
    )

    teks_input = st.text_area(
        'Teks ulasan:',
        placeholder='Contoh: Aplikasi GoPay sangat mudah digunakan, transfer cepat dan aman...',
        height=120
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        tombol = st.button('🔎 Prediksi', use_container_width=True)

    if tombol:
        if not teks_input.strip():
            st.warning('Masukkan teks terlebih dahulu.')
        else:
            with st.spinner('Memproses...'):
                try:
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    teks_bersih = preprocessing_prediksi(teks_input)
                    X_input     = tfidf.transform([teks_bersih])
                    prediksi    = model.predict(X_input)[0]
                    proba       = model.predict_proba(X_input)[0]
                    classes     = model.classes_

                    # Hasil prediksi
                    warna_label = WARNA[prediksi]
                    st.markdown('---')
                    st.markdown(f'### Hasil Prediksi')

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric('Sentimen', EMOJI[prediksi])
                    with col_b:
                        st.metric('Teks asli', f'{len(teks_input.split())} kata')
                    with col_c:
                        st.metric('Teks setelah preprocessing', f'{len(teks_bersih.split())} kata')

                    # Probabilitas per kelas
                    st.markdown('#### Probabilitas per kelas')
                    df_proba = pd.DataFrame({
                        'Kelas'      : [c.capitalize() for c in classes],
                        'Probabilitas': [f'{p*100:.2f}%' for p in proba]
                    })

                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    bars = ax.barh(
                        [c.capitalize() for c in classes],
                        proba * 100,
                        color=[WARNA[c] for c in classes],
                        alpha=0.85, edgecolor='white'
                    )
                    for bar, val in zip(bars, proba * 100):
                        ax.text(
                            bar.get_width() + 0.5,
                            bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', fontsize=10
                        )
                    ax.set_xlabel('Probabilitas (%)')
                    ax.set_xlim(0, 115)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Detail preprocessing
                    with st.expander('Lihat detail preprocessing'):
                        st.markdown(f'**Teks asli:** {teks_input}')
                        st.markdown(f'**Teks bersih:** {teks_bersih}')

                except Exception as e:
                    st.error(f'Error saat prediksi: {e}')

# ============================================================
# HALAMAN 2 — HASIL ANALISIS
# ============================================================
elif halaman == '📊 Hasil Analisis':
    st.title('📊 Hasil Analisis Sentimen')
    st.markdown('Periode data: **Januari – Maret 2026** | Total ulasan: **{:,}**'.format(len(df)))

    # ── Metrik ringkasan ────────────────────────────────────
    st.markdown('---')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Ulasan', f'{len(df):,}')
    with col2:
        n_pos = df[df['label']=='positif'].shape[0]
        st.metric('😊 Positif', f'{n_pos:,}', f'{n_pos/len(df)*100:.1f}%')
    with col3:
        n_neg = df[df['label']=='negatif'].shape[0]
        st.metric('😠 Negatif', f'{n_neg:,}', f'{n_neg/len(df)*100:.1f}%')
    with col4:
        n_net = df[df['label']=='netral'].shape[0]
        st.metric('😐 Netral', f'{n_net:,}', f'{n_net/len(df)*100:.1f}%')

    st.markdown('---')

    # ── Pie chart + Tren ────────────────────────────────────
    col_kiri, col_kanan = st.columns(2)

    with col_kiri:
        st.markdown('#### Distribusi Sentimen')
        dist    = df['label'].value_counts()
        sizes   = [dist[l] for l in LABEL_ORDER]
        colors  = [WARNA[l] for l in LABEL_ORDER]
        explode = (0.03, 0.03, 0.08)

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct='%1.1f%%',
            colors=colors, explode=explode,
            startangle=140, pctdistance=0.78,
            wedgeprops=dict(linewidth=1.5, edgecolor='white')
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_color('white')
            at.set_fontweight('bold')
        legend_labels = [
            f'{l.capitalize()} ({v:,})'
            for l, v in zip(LABEL_ORDER, sizes)
        ]
        ax.legend(
            wedges, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, -0.1),
            ncol=3, fontsize=9
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_kanan:
        st.markdown('#### Tren Sentimen per Bulan')
        tren = (
            df.groupby(['bulan', 'label'])
            .size().reset_index(name='jumlah')
        )
        tren_pivot = (
            tren.pivot(index='bulan', columns='label', values='jumlah')
            .fillna(0).sort_index()
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        for kelas in LABEL_ORDER:
            if kelas in tren_pivot.columns:
                ax.plot(
                    tren_pivot.index, tren_pivot[kelas],
                    marker='o', linewidth=2.5, markersize=7,
                    color=WARNA[kelas], label=kelas.capitalize()
                )
                for x, y in zip(tren_pivot.index, tren_pivot[kelas]):
                    ax.annotate(
                        f'{int(y):,}', (x, y),
                        textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=8, color=WARNA[kelas]
                    )
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Jumlah Ulasan')
        ax.legend(loc='upper right', fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
        )
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Top 15 kata ─────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Top 15 Kata per Kelas Sentimen')

    kelas_dipilih = st.selectbox(
        'Pilih kelas:',
        ['Positif', 'Negatif', 'Netral'],
        label_visibility='collapsed'
    )
    kelas_key = kelas_dipilih.lower()

    semua_kata = ' '.join(
        df[df['label'] == kelas_key]['teks_bersih'].dropna().tolist()
    ).split()
    freq      = Counter(semua_kata).most_common(15)
    kata, jml = zip(*freq)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        list(reversed(kata)), list(reversed(jml)),
        color=WARNA[kelas_key], alpha=0.85, edgecolor='white'
    )
    for bar, val in zip(bars, list(reversed(jml))):
        ax.text(
            bar.get_width() + max(jml)*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9
        )
    ax.set_xlabel('Frekuensi')
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Performa model ───────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Performa Model Naive Bayes')
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric('Accuracy', '84.52%')
    with col_m2:
        st.metric('F1 Positif', '0.91')
    with col_m3:
        st.metric('F1 Negatif', '0.81')
    with col_m4:
        st.metric('F1 Netral', '0.13')

    st.info(
        '**Catatan:** F1-score kelas netral rendah (0.13) karena jumlah data '
        'netral sangat terbatas (4,2% dari total dataset) dan secara linguistik '
        'ulasan netral bersifat ambigu. Hal ini merupakan keterbatasan penelitian '
        'yang telah diantisipasi.'
    )

# ============================================================
# HALAMAN 3 — TOPIK LDA
# ============================================================
elif halaman == '📋 Topik LDA':
    st.title('📋 Pemodelan Topik LDA')
    st.markdown(
        'Topik ditemukan menggunakan **Latent Dirichlet Allocation (LDA)** '
        'yang dijalankan terpisah per kelas sentimen. '
        'Jumlah topik optimal ditentukan berdasarkan **coherence score (c_v)** tertinggi.'
    )

    # ── Tabel coherence score ────────────────────────────────
    st.markdown('---')
    st.markdown('#### Coherence Score per Jumlah Topik')

    try:
        df_coh = pd.read_csv('coherence_scores.csv', encoding='utf-8-sig')
        df_coh.columns = ['k', 'Positif', 'Negatif', 'Netral']

        col_tabel, col_info = st.columns([2, 1])
        with col_tabel:
            st.dataframe(
                df_coh.style.highlight_max(
                    subset=['Positif', 'Negatif', 'Netral'],
                    color='#d4edda'
                ),
                use_container_width=True
            )
        with col_info:
            st.success(
                '**K terbaik:**\n\n'
                '- Positif: k=8 (0.4100)\n'
                '- Negatif: k=8 (0.5151)\n'
                '- Netral: k=8 (0.4310)'
            )
            st.markdown(
                '_Nilai coherence > 0.4 sudah dianggap '
                'cukup untuk data bahasa Indonesia._'
            )
    except:
        st.warning('File coherence_scores.csv tidak ditemukan.')

    # ── Tabel topik per kelas ────────────────────────────────
    st.markdown('---')
    st.markdown('#### Topik per Kelas Sentimen')

    tab_pos, tab_neg, tab_net = st.tabs(['😊 Positif', '😠 Negatif', '😐 Netral'])

    for tab, kelas in zip(
        [tab_pos, tab_neg, tab_net],
        ['positif', 'negatif', 'netral']
    ):
        with tab:
            subset = df_topik[df_topik['kelas'] == kelas][
                ['topik_id', 'nama_topik', 'kata_kunci']
            ].rename(columns={
                'topik_id'  : 'No.',
                'nama_topik': 'Nama Topik',
                'kata_kunci': 'Kata Kunci'
            })
            st.dataframe(subset, use_container_width=True, hide_index=True)

    # ── Interpretasi temuan ──────────────────────────────────
    st.markdown('---')
    st.markdown('#### Integrasi Temuan Naive Bayes + LDA')

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"""
            <div style='background:#E3F2FD; padding:16px; border-radius:10px;
            border-left:4px solid #2196F3;'>
            <b style='color:#2196F3'>😊 Positif — 66.1%</b><br><br>
            Topik dominan: <b>Kemudahan & Kepuasan Transaksi</b>
            dan <b>Keamanan & Kecepatan Proses</b>.<br><br>
            Pengguna puas dengan kemudahan penggunaan,
            kecepatan transaksi, dan rasa aman dalam bertransaksi.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"""
            <div style='background:#FFEBEE; padding:16px; border-radius:10px;
            border-left:4px solid #F44336;'>
            <b style='color:#F44336'>😠 Negatif — 29.7%</b><br><br>
            Topik dominan: <b>Kehilangan Saldo & Gangguan Transfer</b>
            dan <b>Kegagalan Top Up & Lambat</b>.<br><br>
            Keluhan utama berpusat pada masalah transaksi keuangan,
            bukan antarmuka aplikasi.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_c:
        st.markdown(
            f"""
            <div style='background:#FFF3E0; padding:16px; border-radius:10px;
            border-left:4px solid #FF9800;'>
            <b style='color:#FF9800'>😐 Netral — 4.2%</b><br><br>
            Topik dominan: <b>Kendala Teknis Aplikasi</b>
            dan <b>Pengalaman Fitur Pinjaman</b>.<br><br>
            Ulasan netral banyak berisi permintaan bantuan
            atau menunggu perbaikan fitur tertentu.
            </div>
            """,
            unsafe_allow_html=True
        )
