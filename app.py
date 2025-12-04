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
    /* Remove default streamlit background */
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

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Graph Explorer", "Dataset Analysis"])

# Sidebar navigation (remove model performance from here)
st.sidebar.markdown("### Navigation")
st.sidebar.info("Use tabs above to:\n- Make predictions\n- Explore player network graph\n- Analyze dataset statistics")



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
        st.error(f"Model files not found! Please run 'python train_model.py' first.")
        st.error(f"Missing file: {e.filename}")
        st.stop()

# Load semuanya
model, le, scaler, df, config = load_saved_model()

# Display stats
stats_cols = config['stats_cols']

# --- SIDEBAR: INPUT STATISTIK ---
st.sidebar.markdown("---")
st.sidebar.header("Input Atribut Pemain")
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
    # --- MAIN PAGE: HASIL ---
    col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Data Input Pemain")
    
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
    if st.button('Prediksi Posisi Sekarang', use_container_width=True):
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
        
        st.markdown("### Logic Reasoning")
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
        st.markdown("### Feature Scores Breakdown")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Attack Score", f"{attack_score:.1f}/100", help="Finishing + Dribbling + Sprint Speed")
            st.metric("Defense Score", f"{defense_score:.1f}/100", help="Strength + Stamina")
        with col_b:
            st.metric("Midfield Score", f"{midfield_score:.1f}/100", help="Short Passing + Stamina")
            st.metric("Speed Score", f"{speed_score:.1f}/100", help="Acceleration + Sprint Speed")
        with col_c:
            st.metric("Technical Score", f"{technical_score:.1f}/100", help="Dribbling + Short Passing")
            st.metric("Similar Player", closest_player['full_name'][:20], help=f"Posisi: {closest_player['primary_position']}")

        # --- VISUALISASI PROBABILITAS ---
        st.markdown("---")
        st.markdown("### Confidence Level Model")
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
    st.markdown("### Informasi Sistem")
    
    # Reload config to get latest values (not cached)
    try:
        with open('model_config.json', 'r') as f:
            current_config = json.load(f)
    except:
        current_config = config
    
    # Model performance in a card
    st.markdown(f"""
    <div style='background-color: #1a6b3d; padding: 1.5rem; border-radius: 8px; color: white; margin-bottom: 1rem;'>
        <div style='font-size: 1.1rem; font-weight: bold;'>Model Information</div>
        <hr style='border-color: rgba(255,255,255,0.2); margin: 0.5rem 0;'>
        <div style='margin: 0.5rem 0;'><strong>Algorithm:</strong> Random Forest</div>
        <div style='margin: 0.5rem 0;'><strong>Test Accuracy:</strong> {current_config['accuracy']:.2%}</div>
        <div style='margin: 0.5rem 0;'><strong>CV Score:</strong> {current_config['cv_score']:.2%}</div>
        <div style='margin: 0.5rem 0;'><strong>Total Features:</strong> {current_config['n_embedding_features'] + len(current_config['stats_cols']) + 10}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Kamus Posisi")
    st.caption("Daftar lengkap posisi yang dikenali sistem")
    
    # Kelompokkan manual UI-nya
    with st.expander("Kiper & Bek (Defenders)", expanded=False):
        st.markdown("- **GK:** Goalkeeper")
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
    
    # Footer info
    st.markdown("---")
    st.caption("**Tips:** Sesuaikan slider di sidebar untuk mengeksplorasi berbagai profil pemain dan melihat bagaimana perubahan atribut mempengaruhi prediksi posisi.")

# ==================================================
# TAB 2: GRAPH EXPLORER
# ==================================================
with tab2:
    st.markdown("### Player Network Graph Explorer")
    st.markdown("Visualisasi network graph pemain berdasarkan similarity embeddings dari Neo4j")
    
    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        st.markdown("#### Filter Settings")
        
        # Filter by position
        selected_positions = st.multiselect(
            "Filter by Position",
            options=sorted(df['primary_position'].unique()),
            default=[]
        )
        
        # Number of players to show
        n_players = st.slider("Number of Players", 10, 100, 30, 5)
        
        # Similarity threshold
        similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.75, 0.05)
        
        graph_type = st.radio(
            "Graph Type",
            ["Position Clusters", "Player Similarity", "Position Hierarchy"]
        )
        
    with col_left:
        # Filter dataframe
        if selected_positions:
            df_filtered = df[df['primary_position'].isin(selected_positions)].head(n_players)
        else:
            df_filtered = df.head(n_players)
        
        if len(df_filtered) < 2:
            st.warning("Please select at least 2 players to visualize the graph.")
        else:
            with st.spinner("Generating interactive graph..."):
                if graph_type == "Position Clusters":
                    # Create position-based network
                    G = nx.Graph()
                    
                    # Add nodes for each player
                    for idx, row in df_filtered.iterrows():
                        G.add_node(
                            row['full_name'],
                            title=f"{row['full_name']}<br>Position: {row['primary_position']}<br>Age: {int(row['age'])}",
                            group=row['primary_position'],
                            value=10
                        )
                    
                    # Add edges between players of same position
                    positions = df_filtered['primary_position'].unique()
                    for pos in positions:
                        players_in_pos = df_filtered[df_filtered['primary_position'] == pos]['full_name'].tolist()
                        for i, p1 in enumerate(players_in_pos):
                            for p2 in players_in_pos[i+1:]:
                                G.add_edge(p1, p2, weight=2)
                
                elif graph_type == "Player Similarity":
                    # Create similarity-based network using embeddings
                    G = nx.Graph()
                    
                    # Get embeddings
                    embeddings = np.array(df_filtered['embedding'].tolist())
                    
                    # Add nodes
                    for idx, row in df_filtered.iterrows():
                        G.add_node(
                            row['full_name'],
                            title=f"{row['full_name']}<br>Position: {row['primary_position']}<br>Age: {int(row['age'])}",
                            group=row['primary_position'],
                            value=10
                        )
                    
                    # Calculate similarity and add edges
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
                
                else:  # Position Hierarchy
                    # Create hierarchical network by position groups
                    G = nx.DiGraph()
                    
                    # Define position hierarchy
                    hierarchy = {
                        'Attack': ['ST', 'CF', 'LW', 'RW'],
                        'Midfield': ['CAM', 'CM', 'CDM', 'LM', 'RM'],
                        'Defense': ['CB', 'LB', 'RB', 'LWB', 'RWB'],
                        'Goalkeeper': ['GK']
                    }
                    
                    # Add category nodes
                    for category in hierarchy.keys():
                        G.add_node(category, title=category, group=category, value=30, shape='box')
                    
                    # Add player nodes and connect to categories
                    for idx, row in df_filtered.iterrows():
                        pos = row['primary_position']
                        player_name = row['full_name']
                        
                        G.add_node(
                            player_name,
                            title=f"{player_name}<br>Position: {pos}<br>Age: {int(row['age'])}",
                            group=pos,
                            value=10
                        )
                        
                        # Connect to appropriate category
                        for category, positions in hierarchy.items():
                            if pos in positions:
                                G.add_edge(category, player_name)
                                break
                
                # Create PyVis network
                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#1a6b3d")
                net.from_nx(G)
                
                # Customize physics
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
                
                # Save and display
                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                        net.save_graph(f.name)
                        temp_file = f.name
                    
                    # Read and display
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    st.components.v1.html(html_content, height=620)
                    
                    # Cleanup
                    os.unlink(temp_file)
                    
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")
                    st.info("Try reducing the number of players or adjusting filters.")
        
        st.markdown("---")
        st.markdown("#### Graph Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total in Dataset", len(df))
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
        
        # Get top positions
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
        
        # Calculate average stats per position
        stats_by_pos = df.groupby('primary_position')[stats_cols].mean()
        
        # Select a position to display
        selected_pos = st.selectbox("Select Position", sorted(df['primary_position'].unique()))
        
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
            
            # Add value labels
            for i, v in enumerate(stats):
                ax.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("#### Dataset Overview")
    
    col6, col7, col8 = st.columns(3)
    # with col5:
    #     st.metric("Total Players", len(df))
    with col6:
        st.metric("Unique Positions", df['primary_position'].nunique())
    with col7:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col8:
        st.metric("Feature Dimensions", config['n_embedding_features'] + len(config['stats_cols']) + 10)
    
    st.markdown("#### Sample Data")
    display_cols = ['full_name', 'positions', 'primary_position'] + stats_cols
    st.dataframe(
        df[display_cols].head(20),
        use_container_width=True
    )