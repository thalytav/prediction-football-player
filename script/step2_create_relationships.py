"""
STEP 2: Create Relationships Between Nodes
===========================================
Script ini membuat relasi antar pemain berdasarkan similarity statistik mereka.
Relasi ini akan digunakan untuk generate embeddings di Step 3.

Relasi yang dibuat:
- SIMILAR_TO: Antar pemain dengan cosine similarity > threshold

Prasyarat:
- Step 1 sudah selesai (players dan positions ada di Neo4j)

Output:
- Relasi SIMILAR_TO antar pemain dengan weight similarity
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


class RelationshipBuilder:
    """Build relationships between player nodes"""
    
    def __init__(self, uri, username, password):
        """Initialize connection"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info(f"✅ Connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
    
    def get_all_players_stats(self):
        """Get all players with their stats"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Player)
            RETURN p.full_name as name,
                   p.age as age,
                   p.acceleration as acceleration,
                   p.sprint_speed as sprint_speed,
                   p.dribbling as dribbling,
                   p.short_passing as short_passing,
                   p.finishing as finishing,
                   p.stamina as stamina,
                   p.strength as strength
            """
            result = session.run(query)
            
            players = []
            for record in result:
                players.append({
                    'name': record['name'],
                    'stats': [
                        float(record['age']),
                        float(record['acceleration']),
                        float(record['sprint_speed']),
                        float(record['dribbling']),
                        float(record['short_passing']),
                        float(record['finishing']),
                        float(record['stamina']),
                        float(record['strength'])
                    ]
                })
            
            logger.info(f"✅ Retrieved {len(players)} players")
            return players
    
    def batch_create_similarity_relationships(self, relationships_batch):
        """Batch create SIMILAR_TO relationships using UNWIND"""
        with self.driver.session() as session:
            session.execute_write(self._create_similarities_batch, relationships_batch)
    
    @staticmethod
    def _create_similarities_batch(tx, batch):
        """Transaction to create multiple SIMILAR_TO relationships"""
        query = """
        UNWIND $batch AS rel
        MATCH (p1:Player {full_name: rel.player1})
        MATCH (p2:Player {full_name: rel.player2})
        MERGE (p1)-[r:SIMILAR_TO]-(p2)
        SET r.similarity = rel.similarity,
            r.weight = rel.similarity,
            r.created_at = datetime()
        """
        tx.run(query, batch=batch)
    
    def delete_existing_similarities(self):
        """Delete existing SIMILAR_TO relationships"""
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:SIMILAR_TO]-() DELETE r")
            logger.info("🗑️  Deleted existing SIMILAR_TO relationships")
    
    def get_relationship_count(self):
        """Count SIMILAR_TO relationships"""
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:SIMILAR_TO]-() RETURN count(r) as count")
            return result.single()["count"]


def calculate_player_similarities(players, threshold=0.7, top_k=10):
    """
    Calculate pairwise similarity between players
    
    Args:
        players: List of player dicts with stats
        threshold: Minimum similarity to create relationship
        top_k: Maximum number of similar players per player
    
    Returns:
        List of (player1, player2, similarity) tuples
    """
    logger.info(f"Calculating similarities (threshold={threshold}, top_k={top_k})")
    
    # Extract names and stats
    names = [p['name'] for p in players]
    stats = np.array([p['stats'] for p in players])
    
    # Normalize stats
    scaler = StandardScaler()
    stats_normalized = scaler.fit_transform(stats)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(stats_normalized)
    
    # Extract relationships
    relationships = []
    
    for i in tqdm(range(len(players)), desc="Processing similarities"):
        # Get similarity scores for player i
        similarities = similarity_matrix[i]
        
        # Sort by similarity (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in similar_indices if idx != i]
        
        # Take top_k most similar players above threshold
        count = 0
        for j in similar_indices:
            sim_score = similarities[j]
            
            if sim_score >= threshold and count < top_k:
                # Avoid duplicate relationships (only create if i < j)
                if i < j:
                    relationships.append((names[i], names[j], float(sim_score)))
                count += 1
            
            if count >= top_k:
                break
    
    logger.info(f"✅ Found {len(relationships)} relationships above threshold")
    return relationships


def create_all_relationships(builder, relationships, batch_size=1000):
    """Create all similarity relationships in optimized batches using UNWIND"""
    logger.info(f"Creating {len(relationships)} relationships in batches of {batch_size}...")
    
    total_batches = (len(relationships) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(relationships), batch_size), total=total_batches, desc="Relationship Batches"):
        batch = relationships[i:i+batch_size]
        
        # Prepare batch data for UNWIND
        batch_data = [
            {
                'player1': player1,
                'player2': player2,
                'similarity': similarity
            }
            for player1, player2, similarity in batch
        ]
        
        try:
            builder.batch_create_similarity_relationships(batch_data)
        except Exception as e:
            logger.error(f"Batch failed at index {i}: {e}")
    
    logger.info(f"✅ Completed creating relationships")
    return len(relationships)  # Return total count


def verify_relationships(builder):
    """Verify relationships were created"""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)
    
    count = builder.get_relationship_count()
    logger.info(f"✅ Total SIMILAR_TO relationships: {count}")
    
    # Sample relationships
    with builder.driver.session() as session:
        result = session.run("""
            MATCH (p1:Player)-[r:SIMILAR_TO]-(p2:Player)
            RETURN p1.full_name as player1, 
                   p2.full_name as player2, 
                   r.similarity as similarity
            ORDER BY r.similarity DESC
            LIMIT 10
        """)
        
        logger.info("\n📊 Top 10 Most Similar Players:")
        for record in result:
            logger.info(f"  - {record['player1']} ↔ {record['player2']}: {record['similarity']:.3f}")
    
    return count


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("STEP 2: CREATE RELATIONSHIPS")
    print("=" * 60 + "\n")
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Relationship parameters
    SIMILARITY_THRESHOLD = 0.65  # Minimum similarity to create relationship (lowered for more connections)
    TOP_K = 20  # Max similar players per player (increased for better graph connectivity)
    
    print(f"Configuration:")
    print(f"  - Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  - Top K similar players: {TOP_K}\n")
    
    try:
        # Connect
        builder = RelationshipBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Ask about clearing existing relationships
        response = input("Delete existing SIMILAR_TO relationships? (yes/no): ").lower()
        if response == 'yes':
            builder.delete_existing_similarities()
        
        # Get all players and their stats
        logger.info("\n📊 Fetching player data...")
        players = builder.get_all_players_stats()
        
        if len(players) == 0:
            logger.error("❌ No players found! Run step1_insert_data_to_neo4j.py first.")
            return
        
        # Calculate similarities
        logger.info("\n🔍 Calculating similarities...")
        relationships = calculate_player_similarities(
            players, 
            threshold=SIMILARITY_THRESHOLD, 
            top_k=TOP_K
        )
        
        # Create relationships in Neo4j
        logger.info("\n🔗 Creating relationships in Neo4j...")
        created = create_all_relationships(builder, relationships)
        
        # Verify
        verify_relationships(builder)
        
        # Close
        builder.close()
        
        print("\n" + "=" * 60)
        print("✅ STEP 2 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"  - Players processed: {len(players)}")
        print(f"  - Relationships created: {created}")
        print(f"  - Avg connections per player: {created * 2 / len(players):.1f}")
        print("\n🎯 Next: Run step3_generate_embeddings.py")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
