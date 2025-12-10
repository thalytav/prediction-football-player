from neo4j import GraphDatabase
import logging

# Setup logging biar enak debugging-nya
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
            
    def insert_player(self, player_data):
        """Simpan player baru beserta embeddingnya ke Neo4j"""
        with self.driver.session() as session:
            # Query Cypher untuk menyimpan Node Player
            query = """
            MERGE (p:Player {full_name: $full_name})
            SET p += $stats,
                p.embedding = $embedding,
                p.predicted_position = $predicted_position,
                p.source = 'prediction_app',
                p.updated_at = datetime()
            RETURN p
            """
            
            # Gabungkan stats fisik dan domain score jadi satu dict properti
            # Pastikan embedding dikonversi jadi list python biasa
            combined_stats = {**player_data['stats'], **player_data['domain_scores']}
            combined_stats['age'] = player_data['age']
            
            params = {
                'full_name': player_data['full_name'],
                'stats': combined_stats,
                'embedding': player_data['embedding'].tolist() if hasattr(player_data['embedding'], 'tolist') else player_data['embedding'],
                'predicted_position': player_data['predicted_position']
            }
            
            try:
                result = session.run(query, **params)
                record = result.single()
                if record:
                    logger.info(f"✅ Sukses simpan player: {player_data['full_name']}")
                    return record.data() # Kembalikan data dictionary
                return None
            except Exception as e:
                logger.error(f"❌ Gagal simpan player: {e}")
                return None

    def create_position_relationship(self, player_name, position_code):
        """Buat relasi PLAYS_AS dari Player ke Position"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Player {full_name: $player_name})
            MERGE (pos:Position {code: $position_code})
            MERGE (p)-[:PLAYS_AS]->(pos)
            """
            try:
                session.run(query, player_name=player_name, position_code=position_code)
                logger.info(f"✅ Relasi PLAYS_AS ke {position_code} berhasil dibuat.")
            except Exception as e:
                logger.error(f"❌ Gagal buat relasi: {e}")

# Helper function yang dipanggil di app.py
def create_neo4j_connection(uri, username, password):
    return Neo4jConnector(uri, username, password)