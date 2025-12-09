# 🚀 Complete Workflow Guide: Data to Prediction

## Overview Lengkap Pipeline

Pipeline ini mengikuti workflow end-to-end yang benar untuk graph-based machine learning:

```
CSV Data → Neo4j → Relationships → Graph Embeddings → Train Model → Predict → Save Back to Neo4j
```

### Tahapan Workflow

#### **STEP 1: Insert Data ke Neo4j**
📂 Script: `step1_insert_data_to_neo4j.py`

**Yang Dilakukan:**
- Membaca data pemain dari `fifa_players.csv` (data mentah/raw)
- Insert semua pemain sebagai node `Player` di Neo4j
- Membuat node `Position` untuk setiap posisi unik
- Membuat relasi `PLAYS_AS` antara pemain dan posisi

**Output:**
- ~5000 player nodes dengan semua atribut
- ~15 position nodes
- ~5000 PLAYS_AS relationships

**Perintah:**
```bash
cd script
python step1_insert_data_to_neo4j.py
```

**Verifikasi di Neo4j Browser:**
```cypher
MATCH (p:Player) RETURN count(p)
MATCH (pos:Position) RETURN pos.code
MATCH (p:Player)-[r:PLAYS_AS]->(pos) RETURN p, r, pos LIMIT 10
```

---

#### **STEP 2: Create Similarity Relationships**
📂 Script: `step2_create_relationships.py`

**Yang Dilakukan:**
- Mengambil semua pemain dan statistik mereka
- Menghitung cosine similarity antar pemain
- Membuat relasi `SIMILAR_TO` untuk pemain yang mirip (similarity > threshold)

**Parameters:**
- Similarity threshold: 0.70 (configurable)
- Top K similar players: 15 per player

**Output:**
- Ribuan relasi `SIMILAR_TO` dengan weight similarity
- Membentuk connected graph yang siap untuk embedding

**Perintah:**
```bash
python step2_create_relationships.py
```

**Verifikasi:**
```cypher
MATCH ()-[r:SIMILAR_TO]-() RETURN count(r)
MATCH (p1)-[r:SIMILAR_TO]-(p2) 
RETURN p1.full_name, p2.full_name, r.similarity 
ORDER BY r.similarity DESC LIMIT 20
```

---

#### **STEP 3: Generate Graph Embeddings**
📂 Script: `step3_generate_embeddings.py`

**Yang Dilakukan:**
- Extract graph structure dari Neo4j ke NetworkX
- Menjalankan Node2Vec algorithm untuk random walks
- Train Word2Vec model untuk generate embeddings
- Menyimpan 64-dimensional embedding untuk setiap player di Neo4j
- Export embeddings + stats ke CSV untuk training

**Algorithm:** Node2Vec
- Dimensions: 64
- Walk length: 30
- Number of walks: 200 per node
- Context window: 10

**Output:**
- Property `embedding` (64D vector) untuk setiap player
- File `player_embeddings.csv` untuk training

**Perintah:**
```bash
python step3_generate_embeddings.py
```

**Verifikasi:**
```cypher
MATCH (p:Player) 
WHERE p.embedding IS NOT NULL 
RETURN p.full_name, size(p.embedding), p.embedding[0..3] 
LIMIT 5
```

---

#### **STEP 4: Train Model dengan Embeddings**
📂 Script: `step4_train_model.py`

**Yang Dilakukan:**
- Load `player_embeddings.csv` yang sudah ada embeddings
- Build 82-feature matrix:
  - 64 embedding dimensions
  - 5 embedding statistics (mean, std, max, min, range)
  - 8 normalized player stats
  - 5 domain features (attack, defense, midfield, speed, technical)
- Apply SMOTE untuk class balance
- Train Random Forest classifier
- Save model, scaler, label encoder, dan config

**Output:**
- `best_football_model.pkl` - Trained model
- `scaler.pkl` - Stats scaler
- `label_encoder.pkl` - Position encoder
- `model_config.json` - Config dan metrics

**Perintah:**
```bash
python step4_train_model.py
```

**Expected Performance:**
- Test Accuracy: ~92%
- Cross-validation Score: ~90%

---

#### **STEP 5: Predict dan Save ke Neo4j**
📂 Script: `step5_predict_and_save.py`

**Yang Dilakukan:**
- Input data pemain baru (nama + 8 stats)
- Cari pemain paling mirip di database untuk ambil embeddingnya
- Build 82 features untuk prediction
- Predict posisi menggunakan trained model
- **Save pemain baru ke Neo4j dengan:**
  - Semua atribut teknis
  - Embedding (dari similar player)
  - Predicted position
  - Domain scores
  - Relationship PLAYS_AS

**Perintah:**
```bash
python step5_predict_and_save.py
```

**Demo:** Script akan memprediksi 3 pemain contoh dan save ke Neo4j

**Verifikasi:**
```cypher
MATCH (p:Player {source: 'prediction'}) 
RETURN p.full_name, p.predicted_position, p.embedding[0..3]
```

---

## 🎯 Quick Start: Run All Steps

### Option A: Manual Step-by-Step
```bash
cd script

# Install dependencies
pip install -r ../requirements.txt

# Setup Neo4j credentials
copy ..\.env.example ..\.env
# Edit .env dengan password Neo4j kamu

# Run each step
python step1_insert_data_to_neo4j.py
python step2_create_relationships.py
python step3_generate_embeddings.py
python step4_train_model.py
python step5_predict_and_save.py
```

### Option B: Run All at Once
```bash
cd script
python run_all_steps.py
```

Script `run_all_steps.py` akan menjalankan semua 5 steps secara otomatis.

---

## 🔧 Prerequisites

### 1. Neo4j Database
- Neo4j Desktop atau Docker container
- URI: `bolt://localhost:7687`
- Username: `neo4j`
- Password: (set di `.env`)

**Start Neo4j (Docker):**
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Files
- `data/cleaned_football_data.csv` - Harus tersedia
- Minimal columns: `full_name`, `age`, `positions`, technical stats (8 cols)

---

## 📊 Data Flow Diagram

```
┌─────────────────┐
│  CSV Dataset    │
│  (5000 players) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 1: Neo4j Insertion            │
│  - Player nodes (with stats)        │
│  - Position nodes                   │
│  - PLAYS_AS relationships           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 2: Similarity Relationships   │
│  - Calculate cosine similarity      │
│  - Create SIMILAR_TO edges          │
│  - Build connected graph            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 3: Graph Embeddings           │
│  - Node2Vec random walks            │
│  - Word2Vec training                │
│  - 64D embeddings per player        │
│  - Save to Neo4j + export CSV       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 4: Train Model                │
│  - Load embeddings CSV              │
│  - Feature engineering (82 features)│
│  - Random Forest + SMOTE            │
│  - Save model artifacts             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 5: Predict & Save             │
│  - Input new player stats           │
│  - Find similar player → embedding  │
│  - Predict position                 │
│  - Save to Neo4j with embedding     │
└─────────────────────────────────────┘
```

---

## 🎮 Menggunakan Streamlit App

Setelah Step 1-4 selesai, kamu bisa menggunakan web interface:

```bash
cd script
streamlit run app.py
```

**Fitur App:**
1. **Prediction Tab:**
   - Input stats pemain baru
   - Predict posisi
   - Otomatis save ke Neo4j (jika enabled)

2. **Graph Explorer:**
   - Visualisasi network graph
   - 3 mode: Position Clusters, Player Similarity, Position Hierarchy

3. **Dataset Analysis:**
   - Position distribution
   - Age statistics
   - Correlation heatmaps

---

## 🔍 Troubleshooting

### Issue: "No players found" di Step 2/3
**Solution:** Jalankan Step 1 terlebih dahulu

### Issue: "Connection refused" ke Neo4j
**Solution:** 
- Pastikan Neo4j running (`http://localhost:7474`)
- Cek credentials di `.env`
- Test koneksi di Neo4j Browser

### Issue: "CSV file not found"
**Solution:**
- Pastikan berada di folder `script/`
- File CSV harus ada di `data/cleaned_football_data.csv`

### Issue: Node2Vec terlalu lambat
**Solution:**
- Kurangi `num_walks` dari 200 → 100
- Kurangi `walk_length` dari 30 → 20
- Gunakan workers lebih banyak (sesuai CPU cores)

---

## 📈 Expected Timeline

| Step | Duration | Can Skip? |
|------|----------|-----------|
| Step 1: Insert Data | 2-5 min | ❌ Required |
| Step 2: Relationships | 3-10 min | ❌ Required |
| Step 3: Embeddings | 5-15 min | ❌ Required |
| Step 4: Train Model | 2-5 min | ❌ Required |
| Step 5: Demo Predict | < 1 min | ✅ Optional |

**Total:** ~15-35 minutes untuk complete pipeline

---

## 🎓 Konsep Penting

### Kenapa Perlu Graph Embeddings?
- **Raw stats alone:** Tidak capture hubungan struktural antar pemain
- **Graph structure:** Pemain mirip terhubung → embedding capture similarity patterns
- **Node2Vec:** Random walks capture neighborhood info → better representations

### Similarity Relationships
- Threshold 0.70: Balance antara terlalu sparse vs terlalu dense graph
- Top K=15: Setiap pemain punya cukup neighbors untuk good embedding

### Feature Engineering (82 features)
1. **Embeddings (64):** Graph structure information
2. **Embedding stats (5):** Statistical summary of embedding
3. **Normalized stats (8):** Atribut teknis player
4. **Domain features (5):** Football-specific aggregations

### Predict New Player
- Tidak punya graph history → ambil embedding dari similar player
- Reasonable assumption: pemain dengan stats mirip punya posisi mirip
- Embedding borrowed mengandung structural info yang relevan

---

## 📝 Notes untuk Demo/Presentation

**Flow yang benar untuk dijelaskan:**
1. "Kami memulai dengan insert 5000 pemain ke Neo4j sebagai graph nodes"
2. "Lalu membuat relasi similarity berdasarkan statistical attributes"
3. "Dari graph structure ini, kami generate embeddings menggunakan Node2Vec"
4. "Embeddings ini capture structural patterns dari similarity network"
5. "Model dilatih dengan kombinasi embeddings dan raw stats → accuracy 92%"
6. "Untuk pemain baru, kami ambil embedding dari pemain paling mirip"
7. "Sistem real-time: predict → langsung save ke Neo4j dengan embedding"

**Keunggulan:**
- ✅ Graph-aware: Embedding capture relational structure
- ✅ Scalable: New players bisa langsung diprediksi
- ✅ Real-time: Instant prediction + database update
- ✅ Complete: Data, model, dan database always in sync

---

## 🤝 Kontribusi

Kelompok 10:
- Thalyta Vius Pramesti (5025231055)
- Winda Nafiqih Irawan (5025231065)
- Miskiyah (5025231119)

**Mata Kuliah:** RSBP - Riset Sistem Basis Data dan Pemrosesan Big Data
**Topik:** Prediksi Posisi Pemain Sepak Bola Menggunakan Graph Database dan Machine Learning

---

Selamat mencoba! 🚀⚽
