"""
STEP 5: Predict dan Save Back to Neo4j
=======================================
Script ini mendemonstrasikan full flow:
1. Input data pemain baru
2. Generate embedding (menggunakan similar player approach)
3. Predict posisi
4. Save ke Neo4j dengan embedding

Prasyarat:
- Step 1-4 sudah selesai
- Model terlatih tersedia

Output:
- Node player baru di Neo4j dengan embedding dan predicted position
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


class PredictorWithNeo4j:
    """Predictor that saves results back to Neo4j"""
    
    def __init__(self, neo4j_uri, username, password, model_dir="../model"):
        """Initialize with Neo4j connection and load model"""
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
            logger.info(f"✅ Connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
        
        # Resolve model directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if model_dir.startswith("../"):
            model_dir = os.path.join(script_dir, model_dir)
        model_dir = os.path.normpath(model_dir)
        
        # Load model artifacts
        logger.info(f"📦 Loading model from {model_dir}...")
        self.model = joblib.load(f"{model_dir}/best_football_model.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.le = joblib.load(f"{model_dir}/label_encoder.pkl")
        
        with open(f"{model_dir}/model_config.json", 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"✅ Model loaded (accuracy: {self.config['accuracy']:.4f})")
        
        # Lazy loading: reference data will be loaded on first use
        self.reference_df = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def load_reference_data(self):
        """Load existing players for similarity matching (lazy loading)"""
        if self.reference_df is not None:
            return  # Already loaded
        
        logger.info("📊 Loading reference data from Neo4j...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player)
                WHERE p.embedding IS NOT NULL
                RETURN p.full_name as name,
                       p.age as age,
                       p.acceleration as acceleration,
                       p.sprint_speed as sprint_speed,
                       p.dribbling as dribbling,
                       p.short_passing as short_passing,
                       p.finishing as finishing,
                       p.stamina as stamina,
                       p.strength as strength,
                       p.primary_position as position,
                       p.embedding as embedding
            """)
            
            data = []
            for record in result:
                data.append({
                    'name': record['name'],
                    'age': record['age'],
                    'acceleration': record['acceleration'],
                    'sprint_speed': record['sprint_speed'],
                    'dribbling': record['dribbling'],
                    'short_passing': record['short_passing'],
                    'finishing': record['finishing'],
                    'stamina': record['stamina'],
                    'strength': record['strength'],
                    'position': record['position'],
                    'embedding': record['embedding']
                })
            
            self.reference_df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(self.reference_df)} reference players")
    
    def find_similar_player(self, input_stats):
        """
        Find most similar player based on stats
        Returns the embedding of the most similar player
        """
        # Ensure reference data is loaded
        self.load_reference_data()
        
        stats_cols = self.config['stats_cols']
        
        # Get stats from reference data
        ref_stats = self.reference_df[stats_cols].values
        
        # Calculate euclidean distance
        distances = euclidean_distances([input_stats], ref_stats)[0]
        
        # Find closest player
        closest_idx = np.argmin(distances)
        closest_player = self.reference_df.iloc[closest_idx]
        
        logger.info(f"🎯 Most similar player: {closest_player['name']} ({closest_player['position']})")
        logger.info(f"   Distance: {distances[closest_idx]:.2f}")
        
        return closest_player['embedding'], closest_player
    
    def build_features_for_prediction(self, input_stats, embedding):
        """Build 82-feature vector for prediction"""
        # 1. Embedding (64D)
        X_embedding = pd.DataFrame([embedding], columns=[f'emb_{i}' for i in range(len(embedding))])
        
        # 2. Embedding statistics (5)
        emb_mean = np.mean(embedding)
        emb_std = np.std(embedding)
        emb_max = np.max(embedding)
        emb_min = np.min(embedding)
        emb_range = emb_max - emb_min
        
        emb_stats = pd.DataFrame([[emb_mean, emb_std, emb_max, emb_min, emb_range]],
                                 columns=['emb_mean', 'emb_std', 'emb_max', 'emb_min', 'emb_range'])
        
        # 3. Normalized stats (8)
        X_stats_scaled = pd.DataFrame(
            self.scaler.transform([input_stats]),
            columns=self.config['stats_cols']
        )
        
        # 4. Domain features (5)
        attack_score = (input_stats[5] + input_stats[3] + input_stats[2]) / 3  # finishing, dribbling, sprint
        defense_score = (input_stats[7] + input_stats[6]) / 2  # strength, stamina
        midfield_score = (input_stats[4] + input_stats[6]) / 2  # short_passing, stamina
        speed_score = (input_stats[1] + input_stats[2]) / 2  # acceleration, sprint
        technical_score = (input_stats[3] + input_stats[4]) / 2  # dribbling, passing
        
        domain_features = pd.DataFrame([[attack_score, defense_score, midfield_score, speed_score, technical_score]],
                                       columns=['attack_score', 'defense_score', 'midfield_score', 
                                               'speed_score', 'technical_score'])
        
        # Combine all
        X_final = pd.concat([
            X_embedding.reset_index(drop=True),
            emb_stats.reset_index(drop=True),
            X_stats_scaled.reset_index(drop=True),
            domain_features.reset_index(drop=True)
        ], axis=1)
        
        return X_final, {
            'attack_score': float(attack_score),
            'defense_score': float(defense_score),
            'midfield_score': float(midfield_score),
            'speed_score': float(speed_score),
            'technical_score': float(technical_score)
        }
    
    def predict_position(self, input_stats):
        """
        Predict position for new player
        
        Args:
            input_stats: List of [age, acc, sprint, dribble, passing, finish, stamina, strength]
        
        Returns:
            predicted_position, confidence, embedding, domain_scores
        """
        # Find similar player and get embedding
        embedding, similar_player = self.find_similar_player(input_stats)
        
        # Build features
        X, domain_scores = self.build_features_for_prediction(input_stats, embedding)
        
        # Predict
        prediction_idx = self.model.predict(X)[0]
        predicted_position = self.le.inverse_transform([prediction_idx])[0]
        
        # Get confidence
        proba = self.model.predict_proba(X)[0]
        confidence = float(proba[prediction_idx])
        
        logger.info(f"🎯 Predicted position: {predicted_position} (confidence: {confidence:.2%})")
        
        return predicted_position, confidence, embedding, domain_scores, similar_player
    
    def save_to_neo4j(self, player_name, input_stats, predicted_position, embedding, domain_scores):
        """Save new player to Neo4j with embedding"""
        logger.info(f"💾 Saving player '{player_name}' to Neo4j...")
        
        with self.driver.session() as session:
            query = """
            MERGE (p:Player {full_name: $full_name})
            SET p.age = $age,
                p.predicted_position = $position,
                p.embedding = $embedding,
                p.acceleration = $acceleration,
                p.sprint_speed = $sprint_speed,
                p.dribbling = $dribbling,
                p.short_passing = $short_passing,
                p.finishing = $finishing,
                p.stamina = $stamina,
                p.strength = $strength,
                p.attack_score = $attack_score,
                p.defense_score = $defense_score,
                p.midfield_score = $midfield_score,
                p.speed_score = $speed_score,
                p.technical_score = $technical_score,
                p.created_at = datetime(),
                p.updated_at = datetime(),
                p.source = 'prediction'
            RETURN p
            """
            
            result = session.run(
                query,
                full_name=player_name,
                age=int(input_stats[0]),
                position=predicted_position,
                embedding=embedding,
                acceleration=int(input_stats[1]),
                sprint_speed=int(input_stats[2]),
                dribbling=int(input_stats[3]),
                short_passing=int(input_stats[4]),
                finishing=int(input_stats[5]),
                stamina=int(input_stats[6]),
                strength=int(input_stats[7]),
                attack_score=domain_scores['attack_score'],
                defense_score=domain_scores['defense_score'],
                midfield_score=domain_scores['midfield_score'],
                speed_score=domain_scores['speed_score'],
                technical_score=domain_scores['technical_score']
            )
            
            record = result.single()
            if record:
                logger.info(f"✅ Player saved successfully!")
                
                # Create PLAYS_AS relationship
                session.run("""
                    MATCH (p:Player {full_name: $player_name})
                    MERGE (pos:Position {code: $position})
                    MERGE (p)-[r:PLAYS_AS]->(pos)
                    SET r.created_at = datetime()
                """, player_name=player_name, position=predicted_position)
                
                logger.info(f"✅ Created PLAYS_AS relationship")
                
                return True
            return False
    
    def predict_and_save(self, player_name, input_stats):
        """
        Full workflow: predict and save to Neo4j
        
        Args:
            player_name: Player's full name
            input_stats: [age, acc, sprint, dribble, passing, finish, stamina, strength]
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"PREDICTING FOR: {player_name}")
        logger.info("=" * 60)
        
        # Predict
        predicted_position, confidence, embedding, domain_scores, similar_player = \
            self.predict_position(input_stats)
        
        # Display results
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"  Player: {player_name}")
        print(f"  Predicted Position: {predicted_position}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Similar to: {similar_player['name']} ({similar_player['position']})")
        print(f"\n  Domain Scores:")
        print(f"    - Attack: {domain_scores['attack_score']:.1f}")
        print(f"    - Defense: {domain_scores['defense_score']:.1f}")
        print(f"    - Midfield: {domain_scores['midfield_score']:.1f}")
        print(f"    - Speed: {domain_scores['speed_score']:.1f}")
        print(f"    - Technical: {domain_scores['technical_score']:.1f}")
        
        # Save to Neo4j
        success = self.save_to_neo4j(player_name, input_stats, predicted_position, 
                                      embedding, domain_scores)
        
        if success:
            print(f"\n✅ Player '{player_name}' successfully saved to Neo4j!")
            return predicted_position, confidence
        else:
            print(f"\n❌ Failed to save player to Neo4j")
            return None, None
    
    def batch_predict_and_save(self, players_data):
        """
        Batch prediction for multiple players
        
        Args:
            players_data: List of dicts with 'name' and 'stats' keys
                         [{'name': 'Player1', 'stats': [25, 85, ...]}, ...]
        
        Returns:
            List of results
        """
        logger.info(f"\n🔄 Batch processing {len(players_data)} players...")
        
        results = []
        for i, player in enumerate(players_data, 1):
            logger.info(f"\n[{i}/{len(players_data)}] Processing {player['name']}...")
            
            position, confidence = self.predict_and_save(player['name'], player['stats'])
            
            if position:
                results.append({
                    'name': player['name'],
                    'position': position,
                    'confidence': confidence,
                    'status': 'success'
                })
            else:
                results.append({
                    'name': player['name'],
                    'status': 'failed'
                })
        
        logger.info(f"\n✅ Batch processing completed: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
        return results


def demo_predictions():
    """Demo dengan beberapa player baru"""
    print("\n" + "=" * 60)
    print("STEP 5: PREDICT AND SAVE TO NEO4J")
    print("=" * 60 + "\n")
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Initialize predictor
    predictor = PredictorWithNeo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    # Demo players
    demo_players = [
        {
            'name': 'John Striker',
            'stats': [25, 85, 90, 75, 70, 92, 80, 75],  # Fast, great finishing - likely ST
            'description': 'Fast striker with excellent finishing'
        },
        {
            'name': 'Mike Defender',
            'stats': [28, 70, 75, 65, 75, 60, 85, 88],  # Strong, good stamina - likely CB
            'description': 'Strong central defender'
        },
        {
            'name': 'Alex Playmaker',
            'stats': [24, 75, 78, 88, 92, 75, 82, 70],  # Great passing and dribbling - likely CAM/CM
            'description': 'Creative midfielder with excellent passing'
        }
    ]
    
    print("Demo: Predicting for 3 new players...\n")
    
    results = []
    for player in demo_players:
        print(f"\n{'='*60}")
        print(f"Player: {player['name']}")
        print(f"Description: {player['description']}")
        print(f"Stats: Age={player['stats'][0]}, Acc={player['stats'][1]}, "
              f"Sprint={player['stats'][2]}, Dribble={player['stats'][3]}")
        print(f"       Pass={player['stats'][4]}, Finish={player['stats'][5]}, "
              f"Stamina={player['stats'][6]}, Strength={player['stats'][7]}")
        
        position, confidence = predictor.predict_and_save(player['name'], player['stats'])
        
        if position:
            results.append({
                'name': player['name'],
                'position': position,
                'confidence': confidence
            })
        
        print(f"{'='*60}\n")
    
    # Close connection
    predictor.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ STEP 5 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📊 Summary of {len(results)} predictions:")
    for r in results:
        print(f"  - {r['name']}: {r['position']} ({r['confidence']:.2%} confidence)")
    
    print("\n🎯 All players saved to Neo4j with embeddings!")
    print("🔍 Verify in Neo4j Browser:")
    print("   MATCH (p:Player {source: 'prediction'}) RETURN p")


if __name__ == "__main__":
    demo_predictions()
