# ⚡ Quick Command Reference

## Setup Awal (Sekali Saja)

```powershell
# 1. Navigate ke project folder
cd C:\college\RSBP\Football\prediction-football-player

# 2. Install semua dependencies
pip install -r requirements.txt

# 3. Setup Neo4j credentials
copy .env.example .env
# Edit .env, isi NEO4J_PASSWORD dengan password Neo4j kamu

# 4. Pastikan Neo4j sudah running
# Cek di browser: http://localhost:7474
```

---

## 🚀 Run Complete Pipeline (Automated)

```powershell
cd script
python run_all_steps.py
```

**Apa yang terjadi:**
- ✅ Insert 5000 players ke Neo4j
- ✅ Create similarity relationships
- ✅ Generate graph embeddings (Node2Vec)
- ✅ Train Random Forest model
- ✅ Demo prediction untuk 3 pemain baru

**Durasi:** 15-35 menit (tergantung hardware)

---

## 📋 Run Step by Step (Manual)

### Step 1: Insert Data ke Neo4j
```powershell
cd script
python step1_insert_data_to_neo4j.py
```
**Output:** Player nodes, Position nodes, PLAYS_AS relationships

**Verify di Neo4j Browser:**
```cypher
MATCH (p:Player) RETURN count(p)
```

---

### Step 2: Create Relationships
```powershell
python step2_create_relationships.py
```
**Output:** SIMILAR_TO relationships berdasarkan cosine similarity

**Verify:**
```cypher
MATCH ()-[r:SIMILAR_TO]-() RETURN count(r)
```

---

### Step 3: Generate Embeddings
```powershell
python step3_generate_embeddings.py
```
**Output:** 64D embeddings untuk setiap player + CSV export

**Verify:**
```cypher
MATCH (p:Player) 
WHERE p.embedding IS NOT NULL 
RETURN count(p)
```

---

### Step 4: Train Model
```powershell
python step4_train_model.py
```
**Output:** Model files di folder `model/`
- `best_football_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `model_config.json`

---

### Step 5: Predict & Save
```powershell
python step5_predict_and_save.py
```
**Output:** 3 demo predictions saved to Neo4j

**Verify:**
```cypher
MATCH (p:Player {source: 'prediction'}) 
RETURN p.full_name, p.predicted_position
```

---

## 🎨 Run Streamlit App

```powershell
cd script
streamlit run app.py
```

**Browser akan buka:** `http://localhost:8501`

**Features:**
- ✅ Prediction dengan Neo4j integration
- ✅ Interactive graph visualization
- ✅ Dataset analysis

---

## 🔍 Useful Neo4j Queries

### Check Total Players
```cypher
MATCH (p:Player) RETURN count(p) as total_players
```

### Check Players with Embeddings
```cypher
MATCH (p:Player) 
WHERE p.embedding IS NOT NULL 
RETURN count(p) as with_embeddings
```

### View Sample Player with Full Details
```cypher
MATCH (p:Player)
RETURN p.full_name, 
       p.age, 
       p.primary_position,
       size(p.embedding) as embedding_dim,
       p.acceleration,
       p.sprint_speed
LIMIT 5
```

### View Network Structure
```cypher
MATCH (p1:Player)-[r:SIMILAR_TO]-(p2:Player)
WHERE p1.full_name = 'Cristiano Ronaldo'
RETURN p1, r, p2
LIMIT 20
```

### View All Positions and Player Count
```cypher
MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
RETURN pos.code as position, 
       count(p) as player_count
ORDER BY player_count DESC
```

### Find Most Similar Players
```cypher
MATCH (p1:Player {full_name: 'Lionel Messi'})-[r:SIMILAR_TO]-(p2:Player)
RETURN p2.full_name, 
       p2.primary_position, 
       r.similarity
ORDER BY r.similarity DESC
LIMIT 10
```

### View Predicted Players
```cypher
MATCH (p:Player {source: 'prediction'})
RETURN p.full_name, 
       p.predicted_position, 
       p.attack_score,
       p.defense_score,
       p.technical_score
```

### Delete All Predicted Players (Cleanup)
```cypher
MATCH (p:Player {source: 'prediction'})
DETACH DELETE p
```

---

## 🛠️ Troubleshooting Commands

### Check Python Version
```powershell
python --version
# Should be 3.8+
```

### List Installed Packages
```powershell
pip list | findstr "neo4j\|node2vec\|streamlit"
```

### Test Neo4j Connection (Python)
```powershell
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your_password')); print('✅ Connected!' if driver.verify_connectivity() else '❌ Failed')"
```

### Check if CSV Exists
```powershell
dir ..\data\cleaned_football_data.csv
```

### View Recent Logs
```powershell
type pipeline_execution.log | Select-Object -Last 50
```

---

## 🔄 Re-run Specific Steps

### Re-insert Data (Clear First)
```powershell
# Di Neo4j Browser, run:
MATCH (n) DETACH DELETE n

# Then:
python step1_insert_data_to_neo4j.py
```

### Re-generate Embeddings Only
```powershell
# Hanya perlu re-run step 3, 4, 5
python step3_generate_embeddings.py
python step4_train_model.py
```

### Re-train Model with Different Parameters
```powershell
# Edit step4_train_model.py, ubah parameters:
# DIMENSIONS = 64  → 128
# NUM_WALKS = 200  → 100

python step4_train_model.py
```

---

## 📊 Performance Optimization

### Faster Embedding Generation
Edit `step3_generate_embeddings.py`:
```python
# Original
NUM_WALKS = 200
WALK_LENGTH = 30

# Faster (masih good quality)
NUM_WALKS = 100
WALK_LENGTH = 20
```

### Use More CPU Cores
```python
WORKERS = 8  # Adjust based on your CPU
```

### Reduce Dataset Size for Testing
Edit `step1_insert_data_to_neo4j.py`:
```python
# Load only first 1000 players
df = df.head(1000)
```

---

## 🎯 Quick Demo for Presentation

### 1. Show Data in Neo4j
```cypher
MATCH (p:Player)-[r:PLAYS_AS]->(pos:Position)
RETURN p, r, pos LIMIT 25
```

### 2. Run Prediction Demo
```powershell
python step5_predict_and_save.py
```

### 3. Verify New Player in Neo4j
```cypher
MATCH (p:Player {source: 'prediction'})
RETURN p
```

### 4. Show Streamlit App
```powershell
streamlit run app.py
```

### 5. Make Live Prediction
- Input nama pemain di sidebar
- Adjust stats dengan sliders
- Enable Neo4j integration
- Click "Prediksi Posisi Sekarang"
- Show result + check Neo4j Browser

---

## 📦 Export/Import Commands

### Export Embeddings CSV
```powershell
# Already done by step3, file at:
dir ..\data\player_embeddings.csv
```

### Backup Neo4j Database
```powershell
# Stop Neo4j first, then:
# Copy data folder dari Neo4j directory
```

### Export Model Files
```powershell
# Model files ada di:
dir ..\model\*.pkl
dir ..\model\model_config.json
```

---

## 🧹 Cleanup Commands

### Clear All Predicted Players
```cypher
MATCH (p:Player {source: 'prediction'})
DETACH DELETE p
```

### Clear All Similarity Relationships
```cypher
MATCH ()-[r:SIMILAR_TO]-()
DELETE r
```

### Clear Entire Database (CAUTION!)
```cypher
MATCH (n)
DETACH DELETE n
```

### Delete Generated Files
```powershell
# Delete embeddings CSV
del ..\data\player_embeddings.csv

# Delete model files
del ..\model\*.pkl
del ..\model\model_config.json

# Delete logs
del pipeline_execution.log
```

---

## ⚙️ Environment Variables (.env)

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

**Load dalam Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("NEO4J_URI")
```

---

## 🎓 For Demo/Presentation

### Complete Demo Flow (5-10 minutes)

```powershell
# Terminal 1: Pastikan Neo4j running
# Browser: http://localhost:7474

# Terminal 2: Run pipeline
cd script
python run_all_steps.py

# Sementara menunggu, show:
# - Architecture diagram
# - Explain graph embeddings concept
# - Show Node2Vec algorithm

# Setelah selesai:
# Terminal 3: Run Streamlit
streamlit run app.py

# Demo di Streamlit:
# 1. Show existing data di Graph Explorer
# 2. Predict pemain baru di Prediction tab
# 3. Check Dataset Analysis

# Neo4j Browser:
MATCH (p:Player {source: 'prediction'})
RETURN p.full_name, p.predicted_position, p.embedding[0..5]

# Show relationship:
MATCH (p:Player {source: 'prediction'})-[r:PLAYS_AS]->(pos)
RETURN p, r, pos
```

---

Semua command siap pakai! Copy-paste aja sesuai kebutuhan. 🚀
