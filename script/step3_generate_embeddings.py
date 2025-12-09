"""
STEP 3: Generate Graph Embeddings
==================================
Script ini menggenerate embeddings dari graph structure yang sudah dibuat.
Menggunakan Node2Vec algorithm untuk random walk based embeddings.

Prasyarat:
- Step 1 dan 2 sudah selesai (nodes dan relationships ada)

Output:
- Embedding 64-dimensional untuk setiap player disimpan di property 'embedding'
- CSV export embeddings untuk training model
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging
import numpy as np
import pandas as pd
from node2vec import Node2Vec
import networkx as nx
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


class EmbeddingGenerator:
    """Generate and save embeddings to Neo4j"""
    
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
    
    def get_graph_structure(self):
        """
        Extract graph structure from Neo4j
        Returns NetworkX graph
        """
        logger.info("📊 Fetching graph structure from Neo4j...")
        
        with self.driver.session() as session:
            # Get all player nodes
            result = session.run("""
                MATCH (p:Player)
                RETURN p.full_name as name,
                       p.age as age,
                       p.acceleration as acceleration,
                       p.sprint_speed as sprint_speed,
                       p.dribbling as dribbling,
                       p.short_passing as short_passing,
                       p.finishing as finishing,
                       p.stamina as stamina,
                       p.strength as strength,
                       p.primary_position as position
            """)
            
            nodes = {}
            for record in result:
                name = record['name']
                nodes[name] = {
                    'position': record['position'],
                    'stats': [
                        record['age'],
                        record['acceleration'],
                        record['sprint_speed'],
                        record['dribbling'],
                        record['short_passing'],
                        record['finishing'],
                        record['stamina'],
                        record['strength']
                    ]
                }
            
            logger.info(f"  ✓ Retrieved {len(nodes)} player nodes")
            
            # Get all relationships
            result = session.run("""
                MATCH (p1:Player)-[r:SIMILAR_TO]-(p2:Player)
                RETURN p1.full_name as player1, 
                       p2.full_name as player2,
                       r.similarity as weight
            """)
            
            edges = []
            for record in result:
                edges.append((
                    record['player1'],
                    record['player2'],
                    record['weight']
                ))
            
            logger.info(f"  ✓ Retrieved {len(edges)} relationships")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for name, attrs in nodes.items():
            G.add_node(name, **attrs)
        
        # Add weighted edges
        for player1, player2, weight in edges:
            if player1 in G and player2 in G:
                G.add_edge(player1, player2, weight=weight)
        
        logger.info(f"✅ Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def generate_node2vec_embeddings(self, G, dimensions=64, walk_length=30, 
                                     num_walks=200, workers=4, window=10, 
                                     min_count=1, batch_words=4):
        """
        Generate Node2Vec embeddings
        
        Args:
            G: NetworkX graph
            dimensions: Embedding dimension
            walk_length: Length of random walk
            num_walks: Number of walks per node
            workers: Number of parallel workers
            window: Context window size
            min_count: Minimum word count
            batch_words: Batch size
        
        Returns:
            Dict mapping node name to embedding vector
        """
        logger.info(f"\n🚀 Generating Node2Vec embeddings...")
        logger.info(f"   Dimensions: {dimensions}")
        logger.info(f"   Walk length: {walk_length}")
        logger.info(f"   Num walks: {num_walks}")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            quiet=False
        )
        
        # Train model
        logger.info("\n📚 Training Word2Vec model...")
        model = node2vec.fit(
            window=window,
            min_count=min_count,
            batch_words=batch_words
        )
        
        # Extract embeddings
        embeddings = {}
        for node in G.nodes():
            try:
                embeddings[node] = model.wv[node].tolist()
            except KeyError:
                # If node not in model, use zero vector
                embeddings[node] = [0.0] * dimensions
                logger.warning(f"  ⚠️  Node '{node}' not in model, using zero vector")
        
        logger.info(f"✅ Generated embeddings for {len(embeddings)} nodes")
        
        return embeddings, model
    
    def batch_save_embeddings_to_neo4j(self, embeddings, batch_size=1000):
        """Save embeddings as node property using batch UNWIND"""
        logger.info("\n💾 Saving embeddings to Neo4j in batches...")
        
        # Prepare batch data
        embedding_data = [
            {'name': player_name, 'embedding': embedding}
            for player_name, embedding in embeddings.items()
        ]
        
        total_batches = (len(embedding_data) + batch_size - 1) // batch_size
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(embedding_data), batch_size), total=total_batches, desc="Saving Batches"):
                batch = embedding_data[i:i+batch_size]
                try:
                    session.execute_write(self._save_embeddings_batch, batch)
                except Exception as e:
                    logger.error(f"Batch failed at index {i}: {e}")
        
        logger.info(f"✅ Saved {len(embedding_data)} embeddings")
        return len(embedding_data)
    
    @staticmethod
    def _save_embeddings_batch(tx, batch):
        """Transaction to save multiple embeddings"""
        query = """
        UNWIND $batch AS item
        MATCH (p:Player {full_name: item.name})
        SET p.embedding = item.embedding,
            p.embedding_updated_at = datetime()
        """
        tx.run(query, batch=batch)
    
    def export_embeddings_to_csv(self, output_path="player_embeddings.csv"):
        """Export embeddings and stats to CSV for model training"""
        # Resolve path relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.normpath(os.path.join(script_dir, '..', 'data', output_path))
        
        logger.info(f"\n📤 Exporting to CSV: {output_path}")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player)
                WHERE p.embedding IS NOT NULL
                RETURN p.full_name as full_name,
                       p.age as age,
                       p.positions as positions,
                       p.primary_position as primary_position,
                       p.acceleration as acceleration,
                       p.sprint_speed as sprint_speed,
                       p.dribbling as dribbling,
                       p.short_passing as short_passing,
                       p.finishing as finishing,
                       p.stamina as stamina,
                       p.strength as strength,
                       p.embedding as embedding
            """)
            
            data = []
            for record in result:
                row = dict(record)
                # Convert embedding list to JSON string for CSV
                row['embedding'] = json.dumps(row['embedding'])
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Exported {len(df)} rows to {output_path}")
            return len(df)
    
    def verify_embeddings(self):
        """Verify embeddings were saved"""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)
        
        with self.driver.session() as session:
            # Count players with embeddings
            result = session.run("""
                MATCH (p:Player)
                WHERE p.embedding IS NOT NULL
                RETURN count(p) as count
            """)
            count = result.single()["count"]
            
            logger.info(f"✅ Players with embeddings: {count}")
            
            # Sample embedding
            result = session.run("""
                MATCH (p:Player)
                WHERE p.embedding IS NOT NULL
                RETURN p.full_name as name, 
                       size(p.embedding) as dim,
                       p.embedding[0..3] as sample
                LIMIT 3
            """)
            
            logger.info("\n📊 Sample Embeddings:")
            for record in result:
                logger.info(f"  - {record['name']}: {record['dim']}D, first 3 values: {record['sample']}")
        
        return count


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE EMBEDDINGS")
    print("=" * 60 + "\n")
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Embedding parameters
    DIMENSIONS = 64
    WALK_LENGTH = 30
    NUM_WALKS = 200
    WORKERS = 4
    
    print(f"Configuration:")
    print(f"  - Embedding dimensions: {DIMENSIONS}")
    print(f"  - Walk length: {WALK_LENGTH}")
    print(f"  - Number of walks: {NUM_WALKS}\n")
    
    try:
        # Connect
        generator = EmbeddingGenerator(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Get graph structure
        G = generator.get_graph_structure()
        
        if G.number_of_nodes() == 0:
            logger.error("❌ No nodes found! Run step1 and step2 first.")
            return
        
        # Generate embeddings
        embeddings, model = generator.generate_node2vec_embeddings(
            G,
            dimensions=DIMENSIONS,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            workers=WORKERS
        )
        
        # Save to Neo4j (batch optimized)
        saved = generator.batch_save_embeddings_to_neo4j(embeddings)
        
        # Export to CSV
        exported = generator.export_embeddings_to_csv()
        
        # Verify
        generator.verify_embeddings()
        
        # Close
        generator.close()
        
        print("\n" + "=" * 60)
        print("✅ STEP 3 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"  - Embeddings generated: {len(embeddings)}")
        print(f"  - Saved to Neo4j: {saved}")
        print(f"  - Exported to CSV: {exported}")
        print(f"  - Embedding dimension: {DIMENSIONS}D")
        print("\n🎯 Next: Run step4_train_model.py")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
