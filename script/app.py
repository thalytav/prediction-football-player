import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from neo4j_connector import create_neo4j_connection
from dotenv import load_dotenv
from neo4j import GraphDatabase  # <-- tambahan untuk koneksi Neo4j langsung

# Load environment variables
load_dotenv()

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="Prediksi Posisi Pemain Bola - FP RSBP",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist green & white football theme
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a6b3d;
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 3px solid #1a6b3d;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #4a4a4a;
        text-align: center;
        padding: 1.5rem 0;
        font-weight: 500;
    }
    
    .metric-card {
        background-color: #f0f9f6;
        border: 2px solid #1a6b3d;
        padding: 1.5rem;
        border-radius: 8px;
        color: #1a6b3d;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1a6b3d;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 6px;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #0f4620;
        box-shadow: 0 2px 8px rgba(26, 107, 61, 0.3);
    }
    
    .prediction-result {
        background-color: #1a6b3d;
        padding: 2rem;
        border-radius: 8px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 3px solid #ffffff;
    }
    
    .info-card {
        background-color: #f0f9f6;
        border-left: 4px solid #1a6b3d;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .section-divider {
        border-top: 2px solid #1a6b3d;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL & INTRO ---
st.markdown('<div class="main-header">AI Football Scout: Position Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediksi posisi ideal pemain sepak bola menggunakan <strong>Graph Database Embeddings</strong>, <strong>Advanced Feature Engineering</strong>, dan <strong>Random Forest</strong></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Graph Explorer", "Dataset Analysis"])

# Sidebar navigation
st.sidebar.markdown("### Navigation")
st.sidebar.info("Use tabs above to:\n- Make predictions\n- Explore player network graph\n- Analyze dataset statistics")

# --- KAMUS POSISI LENGKAP ---
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
    "CF": "Center Forward (Penyerang Lubang / Second Striker)",
    "ST": "Striker (Ujung Tombak)"
}

# --- FUNGSI LOAD DATA & MODEL (DI-CACHE) ---
@st.cache_resource
def load_saved_model():
    """Load pre-trained 15-position model and artifacts"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.normpath(os.path.join(script_dir, '..', 'model'))
        data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))
        
        model = joblib.load(os.path.join(model_dir, 'best_football_model.pkl'))
        le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        df = pd.read_csv(os.path.join(data_dir, 'player_embeddings.csv'))
        df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
        
        return model, le, scaler, df, config
    except FileNotFoundError as e:
        st.error("Model files for 15-class position not found! Please run step4_train_model.py first.")
        st.error(f"Missing file: {e.filename}")
        st.stop()

@st.cache_resource
def load_role_model():
    """Load pre-trained 4-role model and artifacts."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_role_dir = os.path.normpath(os.path.join(script_dir, '..', 'model_role'))

        model_role = joblib.load(os.path.join(model_role_dir, 'best_football_model.pkl'))
        le_role = joblib.load(os.path.join(model_role_dir, 'label_encoder.pkl'))
        scaler_role = joblib.load(os.path.join(model_role_dir, 'scaler.pkl'))

        config_role_path = os.path.join(model_role_dir, 'model_config.json')
        with open(config_role_path, 'r') as f:
            config_role = json.load(f)

        return model_role, le_role, scaler_role, config_role
    except FileNotFoundError as e:
        st.error("Model files for 4-role classifier not found! Please run step4_train_model.py first (model_role).")
        st.error(f"Missing file: {e.filename}")
        st.stop()


def fetch_players_from_neo4j_for_graph(
    uri,
    username,
    password,
    n_players,
    selected_positions,
    focus_name=None
):
    """
    Ambil data pemain untuk Graph Explorer.
    Akan selalu berusaha memasukkan pemain dengan nama = focus_name
    (misalnya pemain yang baru saja diprediksi), jika ada.
    """
    if selected_positions is None:
        selected_positions = []
    
    driver = GraphDatabase.driver(uri, auth=(username, password))

    query_main = """
    MATCH (p:Player)
    WHERE p.embedding IS NOT NULL
      AND (
        size($positions) = 0 OR
        coalesce(p.predicted_position, p.primary_position, "") IN $positions
      )
    RETURN
      p.full_name AS full_name,
      coalesce(p.predicted_position, p.primary_position) AS primary_position,
      p.age AS age,
      p.embedding AS embedding
    ORDER BY full_name
    LIMIT $limit
    """

    # Query khusus untuk pemain fokus (yang baru diprediksi)
    query_focus = """
    MATCH (p:Player {full_name: $focus_name})
    WHERE p.embedding IS NOT NULL
    RETURN
      p.full_name AS full_name,
      coalesce(p.predicted_position, p.primary_position) AS primary_position,
      p.age AS age,
      p.embedding AS embedding
    LIMIT 1
    """

    try:
        with driver.session() as session:
            records = session.run(
                query_main,
                positions=selected_positions,
                limit=int(n_players)
            ).data()

            if focus_name:
                focus_rec = session.run(
                    query_focus,
                    focus_name=focus_name
                ).data()
                if focus_rec:
                    records.extend(focus_rec)
    finally:
        driver.close()
    
    if not records:
        return pd.DataFrame(columns=['full_name', 'primary_position', 'age', 'embedding'])
    
    df_neo = pd.DataFrame(records)

    # Buang duplikat, tapi kita ingin pemain fokus diprioritaskan
    if focus_name:
        # pisahkan baris fokus & lainnya
        mask_focus = df_neo["full_name"] == focus_name
        focus_rows = df_neo[mask_focus]
        other_rows = df_neo[~mask_focus].drop_duplicates(subset="full_name")

        # gabungkan: pemain fokus di atas, baru yang lain
        df_neo = pd.concat([focus_rows, other_rows], ignore_index=True)
    else:
        df_neo = df_neo.drop_duplicates(subset="full_name")

    # Sekarang potong sesuai n_players (pemain fokus sudah di atas jadi tidak terbuang)
    df_neo = df_neo.head(int(n_players))

    df_neo['age'] = pd.to_numeric(df_neo['age'], errors='coerce')
    return df_neo


# Load models & data
model, le, scaler, df, config = load_saved_model()
model_role, le_role, scaler_role, config_role = load_role_model()
stats_cols = config['stats_cols']

# --- SIDEBAR INPUT ---
st.sidebar.markdown("---")
st.sidebar.header("Input Data Pemain Baru")
st.sidebar.markdown("Masukkan informasi pemain untuk prediksi dan penyimpanan ke Neo4j:")

st.sidebar.markdown("#### Identitas Pemain")
player_name = st.sidebar.text_input("Nama Lengkap Pemain", placeholder="e.g., Cristiano Ronaldo")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Neo4j Connection (Optional)")
with st.sidebar.expander("Configure Neo4j Connection"):
    enable_neo4j = st.checkbox("Enable Neo4j Integration", value=False)
    neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username = st.text_input("Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password = st.text_input("Password", type="password", value=os.getenv("NEO4J_PASSWORD", ""))

st.sidebar.markdown("---")
st.sidebar.header("Atribut Pemain")
st.sidebar.markdown("Sesuaikan statistik pemain di bawah ini untuk mendapatkan prediksi posisi:")

def user_input_features():
    st.sidebar.markdown("#### Atribut Fisik")
    age = st.sidebar.slider('Usia (Age)', 15, 45, 25)
    acc = st.sidebar.slider('Acceleration', 0, 100, 75)
    sprint = st.sidebar.slider('Sprint Speed', 0, 100, 75)
    stamina = st.sidebar.slider('Stamina', 0, 100, 70)
    strength = st.sidebar.slider('Strength', 0, 100, 70)
    
    st.sidebar.markdown("#### Atribut Teknik")
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

# ==================================================
# TAB 1: PREDICTION
# ==================================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Data Input Pemain")
        
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
        if st.button('Prediksi Posisi Sekarang', use_container_width=True):
            if enable_neo4j and not player_name.strip():
                st.error("❌ Nama pemain harus diisi untuk menyimpan ke Neo4j!")
                st.stop()
            
            # 1. Cari pemain paling mirip berdasarkan stats
            df_stats = df[stats_cols].fillna(df[stats_cols].median())
            dists = euclidean_distances(input_df, df_stats)
            closest_idx = np.argmin(dists)
            closest_player = df.iloc[closest_idx]
            
            # 2. FEATURE COMMON (embedding + stats + domain)
            player_embedding = np.array(closest_player['embedding']).reshape(1, -1)
            X_embedding = pd.DataFrame(
                player_embedding,
                columns=[f'emb_{i}' for i in range(player_embedding.shape[1])]
            )
            
            emb_mean = np.mean(player_embedding)
            emb_std = np.std(player_embedding)
            emb_max = np.max(player_embedding)
            emb_min = np.min(player_embedding)
            emb_range = emb_max - emb_min
            emb_stats = pd.DataFrame(
                [[emb_mean, emb_std, emb_max, emb_min, emb_range]],
                columns=['emb_mean', 'emb_std', 'emb_max', 'emb_min', 'emb_range']
            )
            
            attack_score = (input_df['finishing'].values[0] +
                            input_df['dribbling'].values[0] +
                            input_df['sprint_speed'].values[0]) / 3
            defense_score = (input_df['strength'].values[0] +
                             input_df['stamina'].values[0]) / 2
            midfield_score = (input_df['short_passing'].values[0] +
                              input_df['stamina'].values[0]) / 2
            speed_score = (input_df['acceleration'].values[0] +
                           input_df['sprint_speed'].values[0]) / 2
            technical_score = (input_df['dribbling'].values[0] +
                               input_df['short_passing'].values[0]) / 2
            
            domain_features = pd.DataFrame(
                [[attack_score, defense_score, midfield_score, speed_score, technical_score]],
                columns=['attack_score', 'defense_score', 'midfield_score', 'speed_score', 'technical_score']
            )
            
            # 3. Stats scaled untuk dua model (15 posisi & 4 role)
            X_stats_scaled_pos = pd.DataFrame(
                scaler.transform(input_df),
                columns=stats_cols
            )
            X_stats_scaled_role = pd.DataFrame(
                scaler_role.transform(input_df),
                columns=stats_cols
            )
            
            # 4. Final feature matrix
            X_final_pos = pd.concat(
                [
                    X_embedding.reset_index(drop=True),
                    emb_stats.reset_index(drop=True),
                    X_stats_scaled_pos.reset_index(drop=True),
                    domain_features.reset_index(drop=True)
                ],
                axis=1
            )
            
            X_final_role = pd.concat(
                [
                    X_embedding.reset_index(drop=True),
                    emb_stats.reset_index(drop=True),
                    X_stats_scaled_role.reset_index(drop=True),
                    domain_features.reset_index(drop=True)
                ],
                axis=1
            )
            
            # 5. Prediksi model 15 posisi
            prediction_idx_pos = model.predict(X_final_pos)[0]
            prediction_label_pos = le.inverse_transform([prediction_idx_pos])[0]
            prediction_proba_pos = model.predict_proba(X_final_pos)[0]
            
            # 6. Prediksi model 4 role
            prediction_idx_role = model_role.predict(X_final_role)[0]
            prediction_label_role = le_role.inverse_transform([prediction_idx_role])[0]
            prediction_proba_role = model_role.predict_proba(X_final_role)[0]
            
            nama_lengkap_posisi = POSISI_LENGKAP.get(prediction_label_pos, prediction_label_pos)
            
            # HEADLINE: 4-role
            st.markdown(f"""
            <div class="prediction-result">
                <div style='font-size: 1rem; opacity: 0.9;'>Recommended Role (4 Kelas)</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{prediction_label_role}</div>
                <div style='font-size: 1.1rem; opacity: 0.9;'>Kelompok peran utama: GK / DEF / MID / FWD</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Posisi spesifik (15 kelas)
            st.markdown("### Prediksi Posisi Spesifik (15 Kelas)")
            st.markdown(f"**Predicted primary position:** `{prediction_label_pos}` — {nama_lengkap_posisi}")
            
            st.markdown("#### Top-5 candidate positions")
            proba_df = pd.DataFrame(prediction_proba_pos, index=le.classes_, columns=['Probability'])
            proba_df = proba_df.sort_values(by='Probability', ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(proba_df)))
            bars = ax.barh(proba_df.index, proba_df['Probability'], color=colors)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.1%}', ha='left', va='center', fontweight='bold', fontsize=10)
            ax.set_xlabel("Probability Score", fontsize=12, fontweight='bold')
            ax.set_ylabel("Position", fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Logic reasoning
            st.markdown("### Logic Reasoning")
            acc_role = config_role.get('accuracy', config_role.get('accuracy_top1', 0.0))
            st.info(f"""
            **Bagaimana sistem membuat prediksi ini?**
            
            1. **Feature Engineering (82 fitur):**
               - 64 dimensi graph embedding dari Neo4j (diambil dari pemain paling mirip: **{closest_player['full_name']}**)
               - 8 atribut statistik yang dinormalisasi
               - 5 statistical features dari embedding
               - 5 domain-specific features (attack, defense, midfield, speed, technical scores)
            
            2. **Role Prediction (4 kelas):**
               - Model khusus **GK/DEF/MID/FWD** dengan akurasi top-1 sekitar **{acc_role:.2%}**
               - Memberi rekomendasi garis besar peran pemain → di sini: **{prediction_label_role}**
            
            3. **Detailed Position Prediction (15 kelas):**
               - Model kedua memetakan ke posisi spesifik seperti ST, LW, CM, CB, dst.
               - Top-1 prediksi: **{prediction_label_pos}**
               - Top-5 alternatif posisi bisa dilihat di grafik probabilitas di atas.
            """)
            
            # Feature scores
            st.markdown("### Feature Scores Breakdown")
            col_a2, col_b2, col_c2 = st.columns(3)
            with col_a2:
                st.metric("Attack Score", f"{attack_score:.1f}/100", help="Finishing + Dribbling + Sprint Speed")
                st.metric("Defense Score", f"{defense_score:.1f}/100", help="Strength + Stamina")
            with col_b2:
                st.metric("Midfield Score", f"{midfield_score:.1f}/100", help="Short Passing + Stamina")
                st.metric("Speed Score", f"{speed_score:.1f}/100", help="Acceleration + Sprint Speed")
            with col_c2:
                st.metric("Technical Score", f"{technical_score:.1f}/100", help="Dribbling + Short Passing")
                st.metric("Similar Player", closest_player['full_name'][:20],
                          help=f"Posisi: {closest_player['primary_position']}")
            
            # Neo4j integration
            if enable_neo4j:
                st.markdown("---")
                st.markdown("### Neo4j Real-Time Integration")
                try:
                    with st.spinner("Connecting to Neo4j..."):
                        neo4j_conn = create_neo4j_connection(
                            uri=neo4j_uri,
                            username=neo4j_username,
                            password=neo4j_password
                        )
                    st.success("✅ Connected to Neo4j successfully!")
                    
                    st.info(f"**Saving Player Data:**\n- Name: **{player_name.strip()}**\n- Age: **{int(input_df['age'].values[0])}**")
                    
                    player_data = {
                        'full_name': player_name.strip(),
                        'age': int(input_df['age'].values[0]),
                        'predicted_position': prediction_label_pos,
                        'predicted_role': prediction_label_role,
                        'embedding': player_embedding.flatten(),
                        'stats': {
                            'acceleration': int(input_df['acceleration'].values[0]),
                            'sprint_speed': int(input_df['sprint_speed'].values[0]),
                            'dribbling': int(input_df['dribbling'].values[0]),
                            'short_passing': int(input_df['short_passing'].values[0]),
                            'finishing': int(input_df['finishing'].values[0]),
                            'stamina': int(input_df['stamina'].values[0]),
                            'strength': int(input_df['strength'].values[0])
                        },
                        'domain_scores': {
                            'attack_score': float(attack_score),
                            'defense_score': float(defense_score),
                            'midfield_score': float(midfield_score),
                            'speed_score': float(speed_score),
                            'technical_score': float(technical_score)
                        }
                    }
                    
                    with st.spinner(f"Inserting player '{player_name}' to Neo4j..."):
                        result = neo4j_conn.insert_player(player_data)
                        
                        if result:
                            st.write(f"✅ Node created: {result.get('full_name')}, Age: {result.get('age')}")
                        else:
                            st.error("❌ Failed to create player node - result is None!")
                            st.stop()
                        
                        neo4j_conn.create_position_relationship(player_name.strip(), prediction_label_pos)
                    
                    st.success(f"✅ Player **{player_name}** successfully saved to Neo4j!")
                    st.info(f"""
                    **Data yang disimpan:**
                    - **Node Type:** Player
                    - **Full Name:** {player_name.strip()}
                    - **Age:** {int(input_df['age'].values[0])}
                    - **Predicted Position:** {prediction_label_pos}
                    - **Predicted Role:** {prediction_label_role}
                    - **Embedding:** 64-dimensional vector
                    - **Technical Stats:** All 8 attributes
                    - **Domain Scores:** 5 calculated features
                    - **Relationship:** PLAYS_AS → {prediction_label_pos}
                    
                    **Verifikasi dengan query:**
                    ```cypher
                    MATCH (p:Player {{full_name: "{player_name.strip()}"}})
                    RETURN p.full_name, p.age, p.predicted_position, p.predicted_role
                    ```
                    """)
                    
                    neo4j_conn.close()
                    # Simpan nama pemain terakhir yang sukses diinsert ke Neo4j
                    st.session_state["last_predicted_player"] = player_name.strip()

                except ConnectionError as e:
                    st.error(f"❌ Failed to connect to Neo4j: {str(e)}")
                    st.warning("Please check your Neo4j credentials and ensure the database is running.")
                except Exception as e:
                    st.error(f"❌ Error inserting to Neo4j: {str(e)}")
                    st.warning("Player prediction completed but Neo4j insertion failed.")

    with col2:
        st.markdown("### Informasi Sistem")

        # ambil config model
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(script_dir, '..', 'model', 'model_config.json'))
            with open(config_path, 'r') as f:
                current_config = json.load(f)
        except:
            current_config = config

        acc_key = "accuracy" if "accuracy" in current_config else "accuracy_top1"
        acc_val = current_config.get(acc_key, 0.0)

        # Model 1 pakai expander
        with st.expander("Model 1 — Primary Position (15 kelas)", expanded=True):
            st.markdown("**Algorithm:** Random Forest")
            st.markdown(f"**Test Accuracy:** {acc_val:.2%}")
            st.markdown(f"**CV Score:** {current_config.get('cv_score', 0.0):.2%}")

        # Model 2 pakai expander
        with st.expander("Model 2 — Role (4 kelas)", expanded=True):
            st.markdown("**Algorithm:** Random Forest")
            st.markdown(
                f"**Test Accuracy:** "
                f"{config_role.get('accuracy', config_role.get('accuracy_top1', 0.0)):.2%}"
            )
            st.markdown(f"**CV Score:** {config_role.get('cv_score', 0.0):.2%}")

        # ----- Kamus Posisi (tetap seperti sebelumnya) -----
        st.markdown("### Kamus Posisi")
        st.caption("Daftar lengkap posisi yang dikenali sistem")

        with st.expander("Kiper (GK)", expanded=False):
            st.markdown("- **GK:** Goalkeeper")

        with st.expander("Bek (Defenders)", expanded=False):
            st.markdown("- **CB:** Center Back")
            st.markdown("- **LB/RB:** Left/Right Back")
            st.markdown("- **LWB/RWB:** Left/Right Wing Back")

        with st.expander("Gelandang (Midfielders)", expanded=False):
            st.markdown("- **CDM:** Central Defensive Midfielder")
            st.markdown("- **CM:** Central Midfielder")
            st.markdown("- **CAM:** Central Attacking Midfielder")
            st.markdown("- **LM/RM:** Left/Right Midfielder")

        with st.expander("Penyerang (Attackers)", expanded=False):
            st.markdown("- **LW/RW:** Left/Right Winger")
            st.markdown("- **CF:** Center Forward")
            st.markdown("- **ST:** Striker")

        st.markdown("---")
        st.caption(
            "**Tips:** Sesuaikan slider di sidebar untuk mengeksplorasi berbagai profil "
            "pemain dan melihat bagaimana perubahan atribut mempengaruhi prediksi posisi."
        )

# ==================================================
# TAB 2: GRAPH EXPLORER
# ==================================================
with tab2:
    st.markdown("### Player Network Graph Explorer")
    st.markdown("Visualisasi network graph pemain berdasarkan similarity embeddings.")

    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        st.markdown("#### Filter Settings")
        
        # Pilih sumber data graf: CSV offline vs Neo4j real-time
        if enable_neo4j:
            data_source = st.radio(
                "Data Source",
                ["Offline CSV", "Neo4j (Real-Time)"],
                index=1
            )
        else:
            data_source = "Offline CSV"
            st.caption("Neo4j integration dimatikan di sidebar → menggunakan dataset CSV offline.")

        selected_positions = st.multiselect(
            "Filter by Position",
            options=sorted(df['primary_position'].dropna().unique()),
            default=[]
        )
        
        n_players = st.slider("Number of Players", 10, 100, 30, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.75, 0.05)
        
        graph_type = st.radio(
            "Graph Type",
            ["Position Clusters", "Player Similarity", "Position Hierarchy"]
        )

        # 🔽 NEW: opsi sorting
        sort_option = st.radio(
            "Sort Nodes By",
            ["Name (A–Z)", "Position", "Random"],
            index=0
        )
        
    with col_left:
        # Tentukan sumber dataframe untuk graf
        if data_source == "Neo4j (Real-Time)" and enable_neo4j:
            try:
                focus_name = st.session_state.get("last_predicted_player")
                with st.spinner("Loading players from Neo4j (real-time)..."):
                    df_source = fetch_players_from_neo4j_for_graph(
                        neo4j_uri,
                        neo4j_username,
                        neo4j_password,
                        n_players,
                        selected_positions if selected_positions else [],
                        focus_name=focus_name
                    )
                if df_source.empty:
                    st.warning("Tidak ada data pemain di Neo4j yang cocok dengan filter. Menggunakan dataset CSV sebagai fallback.")
                    df_source = df.copy()
            except Exception as e:
                st.error(f"Gagal mengambil data dari Neo4j: {e}")
                st.info("Menggunakan dataset CSV offline sebagai fallback.")
                df_source = df.copy()
        else:
            df_source = df.copy()
        
        # 🔽 NEW: terapkan sorting ke df_source
        if not df_source.empty:
            if sort_option == "Name (A–Z)":
                df_source = df_source.sort_values(by="full_name")
            elif sort_option == "Position":
                df_source = df_source.sort_values(by=["primary_position", "full_name"])
            else:  # Random
                df_source = df_source.sample(frac=1, random_state=None).reset_index(drop=True)
        
        # Filter posisi + limit jumlah pemain
        if selected_positions:
            df_filtered = df_source[df_source['primary_position'].isin(selected_positions)].head(n_players)
        else:
            df_filtered = df_source.head(n_players)
        
        if len(df_filtered) < 2:
            st.warning("Please select at least 2 players to visualize the graph.")
        else:
            with st.spinner("Generating interactive graph..."):
                if graph_type == "Position Clusters":
                    G = nx.Graph()
                    for idx, row in df_filtered.iterrows():
                        G.add_node(
                            row['full_name'],
                            title=f"{row['full_name']}<br>Position: {row['primary_position']}<br>Age: {int(row['age']) if not pd.isna(row['age']) else 'N/A'}",
                            group=row['primary_position'],
                            value=10
                        )
                    positions = df_filtered['primary_position'].unique()
                    for pos in positions:
                        players_in_pos = df_filtered[df_filtered['primary_position'] == pos]['full_name'].tolist()
                        for i, p1 in enumerate(players_in_pos):
                            for p2 in players_in_pos[i+1:]:
                                G.add_edge(p1, p2, weight=2)
                
                elif graph_type == "Player Similarity":
                    G = nx.Graph()
                    embeddings = np.array(df_filtered['embedding'].tolist())
                    for idx, row in df_filtered.iterrows():
                        G.add_node(
                            row['full_name'],
                            title=f"{row['full_name']}<br>Position: {row['primary_position']}<br>Age: {int(row['age']) if not pd.isna(row['age']) else 'N/A'}",
                            group=row['primary_position'],
                            value=10
                        )
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(embeddings)
                    for i in range(len(df_filtered)):
                        for j in range(i+1, len(df_filtered)):
                            sim = similarities[i][j]
                            if sim > similarity_threshold:
                                G.add_edge(
                                    df_filtered.iloc[i]['full_name'],
                                    df_filtered.iloc[j]['full_name'],
                                    weight=float(sim),
                                    title=f"Similarity: {sim:.2f}"
                                )
                
                else:
                    G = nx.DiGraph()
                    hierarchy = {
                        'Attack': ['ST', 'CF', 'LW', 'RW'],
                        'Midfield': ['CAM', 'CM', 'CDM', 'LM', 'RM'],
                        'Defense': ['CB', 'LB', 'RB', 'LWB', 'RWB'],
                        'Goalkeeper': ['GK']
                    }
                    for category in hierarchy.keys():
                        G.add_node(category, title=category, group=category, value=30, shape='box')
                    for idx, row in df_filtered.iterrows():
                        pos = row['primary_position']
                        player_name = row['full_name']
                        G.add_node(
                            player_name,
                            title=f"{player_name}<br>Position: {pos}<br>Age: {int(row['age']) if not pd.isna(row['age']) else 'N/A'}",
                            group=pos,
                            value=10
                        )
                        for category, positions in hierarchy.items():
                            if pos in positions:
                                G.add_edge(category, player_name)
                                break
                
                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#1a6b3d")
                net.from_nx(G)
                net.set_options("""
                {
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "centralGravity": 0.3,
                            "springLength": 150,
                            "springConstant": 0.04
                        },
                        "minVelocity": 0.75
                    },
                    "nodes": {
                        "font": {
                            "size": 14,
                            "color": "#1a6b3d"
                        },
                        "borderWidth": 2,
                        "borderWidthSelected": 4
                    },
                    "edges": {
                        "color": {
                            "color": "#cccccc",
                            "highlight": "#1a6b3d"
                        },
                        "smooth": {
                            "type": "continuous"
                        }
                    }
                }
                """)
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                        net.save_graph(f.name)
                        temp_file = f.name
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=620)
                    os.unlink(temp_file)
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")
                    st.info("Try reducing the number of players or adjusting filters.")
        
        st.markdown("---")
        st.markdown("#### Graph Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if data_source == "Neo4j (Real-Time)" and enable_neo4j:
                st.metric("Total (Neo4j sample)", len(df_source))
            else:
                st.metric("Total in Dataset (CSV)", len(df))
        with col_b:
            st.metric("Displayed Players", len(df_filtered))
        with col_c:
            st.metric("Unique Positions", df_filtered['primary_position'].nunique())
        with col_d:
            if len(df_filtered) >= 2:
                st.metric("Graph Nodes", G.number_of_nodes())

# ==================================================
# TAB 3: DATASET ANALYSIS
# ==================================================
with tab3:
    st.markdown("### Dataset Analysis & Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Position Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        position_counts = df['primary_position'].value_counts()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(position_counts)))
        position_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel("Position", fontweight='bold')
        ax.set_ylabel("Count", fontweight='bold')
        ax.set_title("Distribution of Player Positions", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### Age Distribution by Position")
        fig, ax = plt.subplots(figsize=(10, 6))
        top_positions = df['primary_position'].value_counts().head(5).index
        df_top = df[df['primary_position'].isin(top_positions)]
        for pos in top_positions:
            ages = df_top[df_top['primary_position'] == pos]['age']
            ax.hist(ages, alpha=0.5, label=pos, bins=15)
        ax.set_xlabel("Age", fontweight='bold')
        ax.set_ylabel("Frequency", fontweight='bold')
        ax.set_title("Age Distribution (Top 5 Positions)", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Attribute Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_cols = ['age', 'acceleration', 'sprint_speed', 'dribbling', 
                     'short_passing', 'finishing', 'stamina', 'strength']
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Attribute Correlation Matrix", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col4:
        st.markdown("#### Average Stats by Position")
        stats_by_pos = df.groupby('primary_position')[stats_cols].mean()
        selected_pos = st.selectbox("Select Position", sorted(df['primary_position'].dropna().unique()))
        if selected_pos in stats_by_pos.index:
            stats = stats_by_pos.loc[selected_pos]
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(stats))
            colors_bar = plt.cm.viridis(stats / 100)
            ax.barh(y_pos, stats, color=colors_bar)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(stats.index)
            ax.set_xlabel('Average Value', fontweight='bold')
            ax.set_title(f'Average Stats for {selected_pos}', fontweight='bold')
            ax.set_xlim(0, 100)
            for i, v in enumerate(stats):
                ax.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("#### Dataset Overview")
    
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("Unique Positions", df['primary_position'].nunique())
    with col7:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col8:
        st.metric("Feature Dimensions", config['n_embedding_features'] + len(config['stats_cols']) + 10)
    
    st.markdown("#### Sample Data")
    display_cols = ['full_name', 'positions', 'primary_position'] + stats_cols
    st.dataframe(df[display_cols].head(20), use_container_width=True)
