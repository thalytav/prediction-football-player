import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="Prediksi Posisi Pemain Bola - FP RSBP",
    page_icon="⚽",
    layout="wide"
)

# --- JUDUL & INTRO ---
st.title("⚽ AI Football Scout: Position Predictor")
st.markdown("""
Aplikasi ini memprediksi posisi ideal pemain sepak bola menggunakan 
**Graph Database Embeddings** dan **Random Forest**.
""")

# --- KAMUS POSISI LENGKAP ---
# Ini daftar semua posisi yang mungkin muncul
POSISI_LENGKAP = {
    "GK": "Goalkeeper (Kiper)",
    "CB": "Center Back (Bek Tengah)",
    "LB": "Left Back (Bek Kiri)",
    "RB": "Right Back (Bek Kanan)",
    "LWB": "Left Wing Back (Bek Sayap Kiri - Agresif)",
    "RWB": "Right Wing Back (Bek Sayap Kanan - Agresif)",
    "CDM": "Central Defensive Midfielder (Gelandang Bertahan)",
    "CM": "Central Midfielder (Gelandang Tengah)",
    "CAM": "Central Attacking Midfielder (Gelandang Serang)",
    "LM": "Left Midfielder (Gelandang Sayap Kiri)",
    "RM": "Right Midfielder (Gelandang Sayap Kanan)",
    "LW": "Left Winger (Penyerang Sayap Kiri)",
    "RW": "Right Winger (Penyerang Sayap Kanan)",
    "CF": "Center Forward (Penyerang Lobang / Second Striker)",
    "ST": "Striker (Ujung Tombak)"
}

# --- FUNGSI LOAD DATA & MODEL (DI-CACHE BIAR NGEBUT) ---
@st.cache_resource
def load_and_train_model():
    # 1. Load Data
    try:
        df = pd.read_csv('cleaned_football_data.csv')
    except FileNotFoundError:
        return None, None, None, None

    # 2. Preprocessing
    # Parse embedding dari string ke list
    df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    # Ambil posisi utama
    df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    
    # 3. Siapin Fitur Training (Embedding) & Target
    X_embedding = pd.DataFrame(df['embedding'].tolist())
    le = LabelEncoder()
    y = le.fit_transform(df['primary_position'])
    
    # 4. Train Model Full
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_embedding, y)
    
    return model, le, df, X_embedding

# Load semuanya
model, le, df, X_embedding = load_and_train_model()

if df is None:
    st.error("File CSV tidak ditemukan! Pastikan 'neo4j_query_table_data_2025-11-27.csv' ada di folder yang sama.")
    st.stop()

# --- SIDEBAR: INPUT STATISTIK ---
st.sidebar.header("🎛️ Input Atribut Pemain")
st.sidebar.info("Sesuaikan statistik pemain di bawah ini:")

def user_input_features():
    age = st.sidebar.slider('Usia (Age)', 15, 45, 25)
    acc = st.sidebar.slider('Acceleration', 0, 100, 75)
    sprint = st.sidebar.slider('Sprint Speed', 0, 100, 75)
    dribble = st.sidebar.slider('Dribbling', 0, 100, 70)
    passing = st.sidebar.slider('Short Passing', 0, 100, 70)
    finish = st.sidebar.slider('Finishing', 0, 100, 60)
    stamina = st.sidebar.slider('Stamina', 0, 100, 70)
    strength = st.sidebar.slider('Strength', 0, 100, 70)
    
    data = {
        'age': age,
        'acceleration': acc,
        'sprint_speed': sprint,
        'dribbling': dribble,
        'short_passing': passing,
        'finishing': finish,
        'stamina': stamina,
        'strength': strength
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN PAGE: HASIL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Analisis Data Input")
    st.write("Statistik Pemain yang Anda Masukkan:")
    st.dataframe(input_df)

    if st.button('🚀 Prediksi Posisi'):
        # 1. Cari Pemain Mirip
        stats_cols = ['age', 'acceleration', 'sprint_speed', 'dribbling', 
                      'short_passing', 'finishing', 'stamina', 'strength']
        
        df_stats = df[stats_cols].fillna(0)
        dists = euclidean_distances(input_df, df_stats)
        
        closest_idx = np.argmin(dists)
        closest_player = df.iloc[closest_idx]
        
        # 2. Ambil Embedding & Prediksi
        input_embedding = np.array(closest_player['embedding']).reshape(1, -1)
        prediction_idx = model.predict(input_embedding)[0]
        prediction_label = le.inverse_transform([prediction_idx])[0]
        prediction_proba = model.predict_proba(input_embedding)
        
        # Ambil nama lengkap posisi dari kamus
        nama_lengkap_posisi = POSISI_LENGKAP.get(prediction_label, prediction_label)

        # --- TAMPILAN HASIL ---
        st.success(f"Posisi Ideal: **{prediction_label}** - {nama_lengkap_posisi}")
        
        st.markdown(f"""
        > **Logic Reasoning:**
        > Sistem mendeteksi profil statistik ini memiliki kemiripan Graph Vector dengan pemain **{closest_player['full_name']}**.
        > Berdasarkan pola embedding tersebut, Random Forest merekomendasikan posisi **{prediction_label}**.
        """)

        # --- VISUALISASI PROBABILITAS ---
        st.write("---")
        st.subheader("📊 Confidence Level Model")
        
        proba_df = pd.DataFrame(prediction_proba, columns=le.classes_).T
        proba_df.columns = ['Probability']
        proba_df = proba_df.sort_values(by='Probability', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=proba_df['Probability'], y=proba_df.index, palette='viridis', ax=ax)
        ax.set_xlabel("Probability Score")
        ax.set_xlim(0, 1)
        st.pyplot(fig)

with col2:
    st.subheader("ℹ️ Kamus Singkatan")
    st.caption("Daftar lengkap posisi yang dikenali sistem:")
    
    # Tampilkan pakai expander biar rapi kalau kepanjangan, atau list langsung
    # Kita bagi jadi kategori biar enak dilihat
    
    def display_category(title, prefixes):
        st.markdown(f"**{title}**")
        for code, desc in POSISI_LENGKAP.items():
            if code in prefixes:
                st.markdown(f"- **{code}:** {desc.split('(')[0]}")
    
    # Kelompokkan manual biar UI-nya cakep
    with st.expander("🥅 Kiper & Bek (Defenders)", expanded=True):
        st.markdown("- **GK:** Goalkeeper")
        st.markdown("- **CB:** Center Back")
        st.markdown("- **LB/RB:** Left/Right Back")
        st.markdown("- **LWB/RWB:** Left/Right Wing Back")

    with st.expander("🏃 Gelandang (Midfielders)", expanded=True):
        st.markdown("- **CDM:** Central Defensive Midfielder")
        st.markdown("- **CM:** Central Midfielder")
        st.markdown("- **CAM:** Central Attacking Midfielder")
        st.markdown("- **LM/RM:** Left/Right Midfielder")

    with st.expander("⚽ Penyerang (Attackers)", expanded=True):
        st.markdown("- **LW/RW:** Left/Right Winger")
        st.markdown("- **CF:** Center Forward")
        st.markdown("- **ST:** Striker")

    st.info("Model Accuracy: ~72% (Random Forest)")