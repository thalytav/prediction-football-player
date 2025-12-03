import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="Prediksi Posisi Pemain Bola - FP RSBP",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #145a8a;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL & INTRO ---
st.markdown('<div class="main-header">⚽ AI Football Scout: Position Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediksi posisi ideal pemain sepak bola menggunakan <strong>Graph Database Embeddings</strong>, <strong>Advanced Feature Engineering</strong>, dan <strong>Random Forest</strong></div>', unsafe_allow_html=True)

# Display model info in sidebar with better styling
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;'>
        <div style='font-size: 1.2rem; font-weight: bold;'>✅ Accuracy</div>
        <div style='font-size: 2rem; font-weight: bold;'>{config['accuracy']:.2%}</div>
        <div style='font-size: 0.9rem; margin-top: 0.5rem;'>CV Score: {config['cv_score']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)
except:
    pass

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
def load_saved_model():
    """Load pre-trained model and artifacts"""
    try:
        # Load model
        model = joblib.load('best_football_model.pkl')
        le = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load config
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        # Load data for reference
        df = pd.read_csv('cleaned_football_data.csv')
        df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
        
        return model, le, scaler, df, config
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please run 'python train_model.py' first.")
        st.error(f"Missing file: {e.filename}")
        st.stop()

# Load semuanya
model, le, scaler, df, config = load_saved_model()

# Display stats
stats_cols = config['stats_cols']

# --- SIDEBAR: INPUT STATISTIK ---
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Input Atribut Pemain")
st.sidebar.markdown("Sesuaikan statistik pemain di bawah ini untuk mendapatkan prediksi posisi:")

def user_input_features():
    st.sidebar.markdown("#### 🏃 Atribut Fisik")
    age = st.sidebar.slider('Usia (Age)', 15, 45, 25)
    acc = st.sidebar.slider('Acceleration', 0, 100, 75)
    sprint = st.sidebar.slider('Sprint Speed', 0, 100, 75)
    stamina = st.sidebar.slider('Stamina', 0, 100, 70)
    strength = st.sidebar.slider('Strength', 0, 100, 70)
    
    st.sidebar.markdown("#### ⚽ Atribut Teknik")
    dribble = st.sidebar.slider('Dribbling', 0, 100, 70)
    passing = st.sidebar.slider('Short Passing', 0, 100, 70)
    finish = st.sidebar.slider('Finishing', 0, 100, 60)
    
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
    st.markdown("### 🔍 Data Input Pemain")
    
    # Display input in a nice table format
    input_display = input_df.T
    input_display.columns = ['Nilai']
    input_display.index = ['Usia', 'Acceleration', 'Sprint Speed', 'Dribbling', 
                           'Short Passing', 'Finishing', 'Stamina', 'Strength']
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(input_display.iloc[:4], use_container_width=True)
    with col_b:
        st.dataframe(input_display.iloc[4:], use_container_width=True)

    st.markdown("---")
    if st.button('🚀 Prediksi Posisi Sekarang', use_container_width=True):
        # === FEATURE ENGINEERING (SAMA DENGAN TRAINING!) ===
        
        # 1. Cari Pemain Mirip berdasarkan stats
        df_stats = df[stats_cols].fillna(df[stats_cols].median())
        dists = euclidean_distances(input_df, df_stats)
        
        closest_idx = np.argmin(dists)
        closest_player = df.iloc[closest_idx]
        
        # 2. Build features dengan cara yang SAMA dengan training
        # A. Embedding dari pemain terdekat
        player_embedding = np.array(closest_player['embedding']).reshape(1, -1)
        X_embedding = pd.DataFrame(player_embedding)
        
        # B. Embedding statistics
        emb_mean = np.mean(player_embedding)
        emb_std = np.std(player_embedding)
        emb_max = np.max(player_embedding)
        emb_min = np.min(player_embedding)
        emb_range = emb_max - emb_min
        emb_stats = pd.DataFrame([[emb_mean, emb_std, emb_max, emb_min, emb_range]], 
                                  columns=['emb_mean', 'emb_std', 'emb_max', 'emb_min', 'emb_range'])
        
        # C. Normalized raw stats (pake scaler yang sama dari training)
        X_stats_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=stats_cols
        )
        
        # D. Domain features
        attack_score = (input_df['finishing'].values[0] + input_df['dribbling'].values[0] + input_df['sprint_speed'].values[0]) / 3
        defense_score = (input_df['strength'].values[0] + input_df['stamina'].values[0]) / 2
        midfield_score = (input_df['short_passing'].values[0] + input_df['stamina'].values[0]) / 2
        speed_score = (input_df['acceleration'].values[0] + input_df['sprint_speed'].values[0]) / 2
        technical_score = (input_df['dribbling'].values[0] + input_df['short_passing'].values[0]) / 2
        
        domain_features = pd.DataFrame([[attack_score, defense_score, midfield_score, speed_score, technical_score]],
                                        columns=['attack_score', 'defense_score', 'midfield_score', 'speed_score', 'technical_score'])
        
        # E. Combine ALL features (82 total)
        X_final = pd.concat([
            X_embedding.reset_index(drop=True),
            emb_stats.reset_index(drop=True),
            X_stats_scaled.reset_index(drop=True),
            domain_features.reset_index(drop=True)
        ], axis=1)
        
        # Fix column names to strings
        X_final.columns = X_final.columns.astype(str)
        
        # 3. Prediksi dengan model
        prediction_idx = model.predict(X_final)[0]
        prediction_label = le.inverse_transform([prediction_idx])[0]
        prediction_proba = model.predict_proba(X_final)
        
        # Ambil nama lengkap posisi dari kamus
        nama_lengkap_posisi = POSISI_LENGKAP.get(prediction_label, prediction_label)

        # --- TAMPILAN HASIL ---
        st.markdown(f"""
        <div class="prediction-result">
            <div style='font-size: 1rem; opacity: 0.9;'>Posisi Ideal Pemain</div>
            <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{prediction_label}</div>
            <div style='font-size: 1.2rem;'>{nama_lengkap_posisi}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧠 Logic Reasoning")
        st.info(f"""
        **Bagaimana sistem membuat prediksi ini?**
        
        1. **Feature Engineering**: Sistem menggunakan 82 features yang terdiri dari:
           - 64 dimensi graph embedding dari Neo4j
           - 8 atribut statistik yang dinormalisasi
           - 5 statistical features dari embedding
           - 5 domain-specific features (attack, defense, midfield, speed, technical scores)
        
        2. **Similar Player Analysis**: Profil statistik Anda memiliki kemiripan tertinggi dengan pemain **{closest_player['full_name']}** yang bermain di posisi {closest_player['primary_position']}.
        
        3. **Model Prediction**: Random Forest classifier dengan accuracy {config['accuracy']:.2%} menganalisis semua features dan merekomendasikan posisi **{prediction_label}** sebagai posisi paling optimal.
        """)
        
        # Show feature scores
        st.markdown("### 📈 Feature Scores Breakdown")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("⚔️ Attack Score", f"{attack_score:.1f}/100", help="Finishing + Dribbling + Sprint Speed")
            st.metric("🛡️ Defense Score", f"{defense_score:.1f}/100", help="Strength + Stamina")
        with col_b:
            st.metric("🎯 Midfield Score", f"{midfield_score:.1f}/100", help="Short Passing + Stamina")
            st.metric("⚡ Speed Score", f"{speed_score:.1f}/100", help="Acceleration + Sprint Speed")
        with col_c:
            st.metric("🎨 Technical Score", f"{technical_score:.1f}/100", help="Dribbling + Short Passing")
            st.metric("👤 Similar Player", closest_player['full_name'][:20], help=f"Posisi: {closest_player['primary_position']}")

        # --- VISUALISASI PROBABILITAS ---
        st.markdown("---")
        st.markdown("### 📊 Confidence Level Model")
        st.caption("Probabilitas untuk 5 posisi teratas berdasarkan analisis model")
        
        proba_df = pd.DataFrame(prediction_proba, columns=le.classes_).T
        proba_df.columns = ['Probability']
        proba_df = proba_df.sort_values(by='Probability', ascending=False).head(5)
        
        # Create better visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(proba_df)))
        bars = ax.barh(proba_df.index, proba_df['Probability'], color=colors)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1%}', 
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel("Probability Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Position", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

with col2:
    st.markdown("### ℹ️ Informasi Sistem")
    
    # Model performance in a card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
        <div style='font-size: 1.1rem; font-weight: bold;'>🤖 Model Information</div>
        <hr style='border-color: rgba(255,255,255,0.3); margin: 0.5rem 0;'>
        <div style='margin: 0.5rem 0;'><strong>Algorithm:</strong> Random Forest</div>
        <div style='margin: 0.5rem 0;'><strong>Test Accuracy:</strong> {config['accuracy']:.2%}</div>
        <div style='margin: 0.5rem 0;'><strong>CV Score:</strong> {config['cv_score']:.2%}</div>
        <div style='margin: 0.5rem 0;'><strong>Total Features:</strong> {config['n_embedding_features'] + len(config['stats_cols']) + 10}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 Kamus Posisi")
    st.caption("Daftar lengkap posisi yang dikenali sistem")
    
    # Kelompokkan manual biar UI-nya cakep
    with st.expander("🥅 Kiper & Bek (Defenders)", expanded=False):
        st.markdown("- **GK:** Goalkeeper")
        st.markdown("- **CB:** Center Back")
        st.markdown("- **LB/RB:** Left/Right Back")
        st.markdown("- **LWB/RWB:** Left/Right Wing Back")

    with st.expander("🏃 Gelandang (Midfielders)", expanded=False):
        st.markdown("- **CDM:** Central Defensive Midfielder")
        st.markdown("- **CM:** Central Midfielder")
        st.markdown("- **CAM:** Central Attacking Midfielder")
        st.markdown("- **LM/RM:** Left/Right Midfielder")

    with st.expander("⚽ Penyerang (Attackers)", expanded=False):
        st.markdown("- **LW/RW:** Left/Right Winger")
        st.markdown("- **CF:** Center Forward")
        st.markdown("- **ST:** Striker")
    
    # Footer info
    st.markdown("---")
    st.caption("💡 **Tips:** Sesuaikan slider di sidebar untuk mengeksplorasi berbagai profil pemain dan melihat bagaimana perubahan atribut mempengaruhi prediksi posisi.")