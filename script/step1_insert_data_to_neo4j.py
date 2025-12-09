"""
STEP 1: Insert Data dari CSV ke Neo4j Sandbox
==============================================
Script ini membaca data pemain dari CSV dan memasukkannya ke Neo4j sebagai node Player.

Prasyarat:
- Neo4j database sudah running
- Kredensial sudah diset di .env
- File fifa_players.csv tersedia (data mentah)

Output:
- Node Player dengan semua atribut di Neo4j
- Node Position untuk setiap posisi unik
"""

import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Neo4jDataInserter:
    """Class untuk insert data pemain ke Neo4j"""
    
    def __init__(self, uri, username, password):
        """Initialize connection"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info(f"✅ Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            logger.info("Connection closed")
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("⚠️  Database cleared!")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        with self.driver.session() as session:
            try:
                # Constraint untuk Player
                session.run("""
                CREATE CONSTRAINT player_name IF NOT EXISTS
                FOR (p:Player) REQUIRE p.full_name IS UNIQUE
                """)
                logger.info("✅ Created constraint: Player.full_name UNIQUE")
                
                # Constraint untuk Position
                session.run("""
                CREATE CONSTRAINT position_code IF NOT EXISTS
                FOR (pos:Position) REQUIRE pos.code IS UNIQUE
                """)
                logger.info("✅ Created constraint: Position.code UNIQUE")
                
            except Exception as e:
                logger.warning(f"⚠️  Constraints may already exist: {e}")
    
    def batch_insert_players(self, players_data, batch_size=1000):
        """Batch insert players using UNWIND for performance"""
        with self.driver.session() as session:
            total_batches = (len(players_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(players_data), batch_size):
                batch = players_data[i:i+batch_size]
                session.execute_write(self._create_players_batch, batch)
                
            return len(players_data)
    
    @staticmethod
    def _create_players_batch(tx, batch_data):
        """Transaction to create multiple player nodes using UNWIND"""
        query = """
        UNWIND $batch AS player
        MERGE (p:Player {full_name: player.full_name})
        SET p.age = player.age,
            p.positions = player.positions,
            p.primary_position = player.primary_position,
            p.acceleration = player.acceleration,
            p.sprint_speed = player.sprint_speed,
            p.dribbling = player.dribbling,
            p.short_passing = player.short_passing,
            p.finishing = player.finishing,
            p.stamina = player.stamina,
            p.strength = player.strength,
            p.overall = player.overall,
            p.potential = player.potential,
            p.value_eur = player.value_eur,
            p.wage_eur = player.wage_eur,
            p.height_cm = player.height_cm,
            p.weight_kg = player.weight_kg,
            p.created_at = datetime(),
            p.updated_at = datetime()
        """
        tx.run(query, batch=batch_data)
    
    def batch_insert_positions(self, position_codes):
        """Batch insert positions using UNWIND"""
        with self.driver.session() as session:
            session.execute_write(self._create_positions_batch, position_codes)
    
    @staticmethod
    def _create_positions_batch(tx, codes):
        """Transaction to create multiple position nodes"""
        query = """
        UNWIND $codes AS code
        MERGE (pos:Position {code: code})
        SET pos.updated_at = datetime()
        """
        tx.run(query, codes=codes)
    
    def batch_create_relationships(self, relationships_data, batch_size=1000):
        """Batch create PLAYS_AS relationships using UNWIND"""
        with self.driver.session() as session:
            for i in range(0, len(relationships_data), batch_size):
                batch = relationships_data[i:i+batch_size]
                session.execute_write(self._create_relationships_batch, batch)
            return len(relationships_data)
    
    @staticmethod
    def _create_relationships_batch(tx, batch_data):
        """Transaction to create multiple PLAYS_AS relationships"""
        query = """
        UNWIND $batch AS rel
        MATCH (p:Player {full_name: rel.player_name})
        MATCH (pos:Position {code: rel.position_code})
        MERGE (p)-[r:PLAYS_AS]->(pos)
        SET r.created_at = datetime()
        """
        tx.run(query, batch=batch_data)
    
    def get_player_count(self):
        """Get total players in database"""
        with self.driver.session() as session:
            result = session.run("MATCH (p:Player) RETURN count(p) as count")
            return result.single()["count"]
    
    def get_position_count(self):
        """Get total positions"""
        with self.driver.session() as session:
            result = session.run("MATCH (pos:Position) RETURN count(pos) as count")
            return result.single()["count"]


def load_and_prepare_data(csv_path):
    """Load data dari CSV dan prepare untuk insertion"""
    logger.info(f"Loading data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} players from CSV")
    
    # Parse primary position
    if 'primary_position' not in df.columns:
        df['primary_position'] = df['positions'].apply(
            lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown'
        )
    
    # Fill missing values
    numeric_cols = ['age', 'acceleration', 'sprint_speed', 'dribbling', 'short_passing', 
                    'finishing', 'stamina', 'strength', 'overall_rating', 'potential', 
                    'value_euro', 'wage_euro', 'height_cm', 'weight_kgs']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def insert_all_data(inserter, df, batch_size=1000):
    """Insert all players and positions using optimized batch operations"""
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZED BATCH DATA INSERTION")
    logger.info("=" * 60)
    
    # Step 1: Insert all unique positions (fast with UNWIND)
    logger.info("\n📍 Step 1: Inserting Positions...")
    unique_positions = df['primary_position'].unique().tolist()
    inserter.batch_insert_positions(unique_positions)
    logger.info(f"✅ Inserted {len(unique_positions)} positions")
    
    # Step 2: Prepare player data
    logger.info(f"\n👤 Step 2: Preparing {len(df)} players for batch insert...")
    players_data = []
    
    for idx, row in df.iterrows():
        try:
            player_data = {
                'full_name': str(row['full_name']),
                'age': int(row['age']),
                'positions': str(row['positions']),
                'primary_position': str(row['primary_position']),
                'acceleration': int(row['acceleration']),
                'sprint_speed': int(row['sprint_speed']),
                'dribbling': int(row['dribbling']),
                'short_passing': int(row['short_passing']),
                'finishing': int(row['finishing']),
                'stamina': int(row['stamina']),
                'strength': int(row['strength']),
                'overall': int(row.get('overall_rating', 70)),
                'potential': int(row.get('potential', 70)),
                'value_eur': float(row.get('value_euro', 0)),
                'wage_eur': float(row.get('wage_euro', 0)),
                'height_cm': float(row.get('height_cm', 180)),
                'weight_kg': float(row.get('weight_kgs', 75))
            }
            players_data.append(player_data)
        except Exception as e:
            logger.error(f"Failed to prepare {row['full_name']}: {e}")
    
    # Batch insert players
    logger.info(f"\n🚀 Inserting {len(players_data)} players in batches of {batch_size}...")
    total_batches = (len(players_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(players_data), batch_size), total=total_batches, desc="Player Batches"):
        batch = players_data[i:i+batch_size]
        inserter.batch_insert_players(batch, batch_size)
    
    logger.info(f"✅ Inserted {len(players_data)} players")
    
    # Step 3: Prepare relationship data
    logger.info("\n🔗 Step 3: Creating PLAYS_AS Relationships...")
    relationships_data = []
    
    for idx, row in df.iterrows():
        relationships_data.append({
            'player_name': str(row['full_name']),
            'position_code': str(row['primary_position'])
        })
    
    # Batch create relationships
    logger.info(f"\n🚀 Creating {len(relationships_data)} relationships in batches of {batch_size}...")
    total_rel_batches = (len(relationships_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(relationships_data), batch_size), total=total_rel_batches, desc="Relationship Batches"):
        batch = relationships_data[i:i+batch_size]
        inserter.batch_create_relationships(batch, batch_size)
    
    logger.info(f"✅ Created {len(relationships_data)} relationships")
    
    return len(players_data), len(unique_positions), len(relationships_data)


def verify_insertion(inserter):
    """Verify data was inserted correctly"""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)
    
    player_count = inserter.get_player_count()
    position_count = inserter.get_position_count()
    
    logger.info(f"✅ Total Players in DB: {player_count}")
    logger.info(f"✅ Total Positions in DB: {position_count}")
    
    # Sample query
    with inserter.driver.session() as session:
        result = session.run("""
            MATCH (p:Player)-[r:PLAYS_AS]->(pos:Position)
            RETURN p.full_name as player, pos.code as position
            LIMIT 5
        """)
        
        logger.info("\n📊 Sample Data:")
        for record in result:
            logger.info(f"  - {record['player']} → {record['position']}")
    
    return player_count, position_count


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("STEP 1: INSERT DATA TO NEO4J")
    print("=" * 60 + "\n")
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    # Resolve CSV path relative to this script file so it works
    # regardless of current working directory when invoked
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.normpath(os.path.join(script_dir, '..', 'data', 'fifa_players.csv'))
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        logger.error(f"❌ CSV file not found: {CSV_PATH}")
        return
    
    # Confirm before clearing database
    print("⚠️  WARNING: This will insert data into Neo4j.")
    response = input("Do you want to CLEAR existing data first? (yes/no): ").lower()
    clear_db = response == 'yes'
    
    try:
        # Connect to Neo4j
        inserter = Neo4jDataInserter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Clear database if requested
        if clear_db:
            logger.info("Clearing existing data...")
            inserter.clear_database()
        
        # Create constraints
        inserter.create_constraints()
        
        # Load data
        df = load_and_prepare_data(CSV_PATH)
        
        # Insert data
        inserted, positions, relationships = insert_all_data(inserter, df)
        
        # Verify
        verify_insertion(inserter)
        
        # Close connection
        inserter.close()
        
        print("\n" + "=" * 60)
        print("✅ STEP 1 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"  - Players inserted: {inserted}")
        print(f"  - Positions created: {positions}")
        print(f"  - Relationships created: {relationships}")
        print("\n🎯 Next: Run step2_create_relationships.py")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
