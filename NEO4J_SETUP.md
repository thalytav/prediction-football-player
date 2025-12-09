# Neo4j Integration Setup Guide

## Prerequisites
- Neo4j Database installed (Desktop or Server)
- Python 3.8+
- All dependencies from requirements.txt installed

## Quick Setup

### 1. Install Neo4j
**Option A: Neo4j Desktop** (Recommended for development)
- Download from: https://neo4j.com/download/
- Create a new database
- Start the database
- Note your credentials (default username: `neo4j`)

**Option B: Docker**
```bash
docker run \
    --name neo4j-football \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

Edit `.env` file:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_actual_password
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Connection
Create a test script `test_neo4j.py`:
```python
from script.neo4j_connector import create_neo4j_connection
from dotenv import load_dotenv
import os

load_dotenv()

try:
    conn = create_neo4j_connection(
        uri=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    print("✅ Neo4j connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

Run the test:
```bash
python test_neo4j.py
```

## Using the Application

### 1. Start the Application
```bash
cd script
streamlit run app.py
```

### 2. Enable Neo4j Integration
In the Streamlit sidebar:
1. Click "Configure Neo4j Connection" expander
2. Check "Enable Neo4j Integration"
3. Verify credentials (loaded from .env by default)

### 3. Add a New Player
1. Enter player name in "Nama Lengkap Pemain" field
2. Adjust technical attributes using sliders
3. Click "Prediksi Posisi Sekarang"
4. System will:
   - Predict the position
   - Display results and confidence scores
   - Automatically insert player data to Neo4j
   - Create PLAYS_AS relationship to position

### 4. Verify in Neo4j Browser
Open Neo4j Browser (http://localhost:7474) and run:
```cypher
// Show all players
MATCH (p:Player) RETURN p LIMIT 25

// Show player with relationships
MATCH (p:Player)-[r:PLAYS_AS]->(pos:Position)
RETURN p, r, pos

// Show newest players
MATCH (p:Player)
RETURN p.full_name, p.predicted_position, p.age, p.created_at
ORDER BY p.created_at DESC
LIMIT 10

// Get player by name
MATCH (p:Player {full_name: "Cristiano Ronaldo"})
RETURN p
```

## Data Schema

### Player Node Properties
- `full_name` (String): Player's full name (unique identifier)
- `age` (Integer): Player's age
- `predicted_position` (String): Predicted position code (e.g., "ST", "CM")
- `embedding` (List[Float]): 64-dimensional embedding vector
- `acceleration`, `sprint_speed`, `dribbling`, `short_passing`, `finishing`, `stamina`, `strength` (Integer): Technical attributes
- `attack_score`, `defense_score`, `midfield_score`, `speed_score`, `technical_score` (Float): Calculated domain scores
- `created_at` (DateTime): Timestamp of node creation
- `updated_at` (DateTime): Timestamp of last update

### Position Node Properties
- `code` (String): Position code (e.g., "ST", "CM")

### Relationships
- `(Player)-[PLAYS_AS]->(Position)`: Indicates predicted position

## Troubleshooting

### Connection Refused
**Problem**: Cannot connect to Neo4j
**Solution**: 
- Verify Neo4j is running
- Check URI format: `bolt://localhost:7687` (not http://)
- Verify port 7687 is not blocked by firewall

### Authentication Failed
**Problem**: Invalid credentials
**Solution**:
- Verify username/password in .env
- Default username is `neo4j`
- Reset password in Neo4j Desktop if needed

### Module Not Found
**Problem**: `ModuleNotFoundError: No module named 'neo4j'`
**Solution**:
```bash
pip install neo4j python-dotenv
```

### Player Already Exists
**Behavior**: System uses MERGE, so it will UPDATE existing player instead of creating duplicate
**Note**: This is intentional - allows updating player data

## Advanced Queries

### Find Similar Players by Embedding
```cypher
// This requires GDS library - for reference only
MATCH (p:Player)
WITH p, p.embedding AS emb
MATCH (other:Player)
WHERE p <> other
WITH p, other, 
     gds.similarity.cosine(p.embedding, other.embedding) AS similarity
WHERE similarity > 0.8
RETURN p.full_name, other.full_name, similarity
ORDER BY similarity DESC
```

### Position Statistics
```cypher
MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
RETURN pos.code AS position, 
       count(p) AS player_count,
       avg(p.age) AS avg_age,
       avg(p.attack_score) AS avg_attack
ORDER BY player_count DESC
```

### Latest Predictions
```cypher
MATCH (p:Player)
RETURN p.full_name AS name,
       p.predicted_position AS position,
       p.attack_score AS attack,
       p.defense_score AS defense,
       p.created_at AS added
ORDER BY p.created_at DESC
LIMIT 20
```

## Security Notes

1. **Never commit .env file** - it contains sensitive credentials
2. **Use strong passwords** for Neo4j in production
3. **Consider authentication tokens** for API access
4. **Restrict network access** to Neo4j ports in production

## Next Steps

1. Explore the Graph Explorer tab in the app to visualize player networks
2. Use Dataset Analysis tab to see statistics
3. Query Neo4j directly for custom analyses
4. Consider adding more attributes or relationships as needed
