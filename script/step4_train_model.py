"""
STEP 4: Train Model with Embeddings
====================================
Script ini melatih model Random Forest menggunakan embeddings dari Neo4j.

Prasyarat:
- Step 3 sudah selesai (embeddings sudah diexport ke CSV)

Output:
- Model terlatih (best_football_model.pkl)
- Scaler dan label encoder
- Model config dengan accuracy metrics
"""

import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_embeddings_from_csv(csv_path="../data/player_embeddings.csv"):
    """Load embeddings and stats from CSV"""
    logger.info(f"📂 Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} players")
    
    # Parse embedding from JSON string
    df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Parse primary position
    df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    
    logger.info(f"📊 Unique positions: {df['primary_position'].nunique()}")
    logger.info(f"📊 Position distribution:\n{df['primary_position'].value_counts()}")
    
    return df


def build_features(df, stats_cols):
    """
    Build feature matrix with:
    - 64D embedding
    - 8 stats
    - 5 embedding statistics
    - 5 domain features
    Total: 82 features
    """
    logger.info("🔧 Building feature matrix...")
    
    # 1. Embedding features (64D)
    embeddings = np.array(df['embedding'].tolist())
    X_embedding = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    
    # 2. Embedding statistics (5)
    emb_mean = np.mean(embeddings, axis=1)
    emb_std = np.std(embeddings, axis=1)
    emb_max = np.max(embeddings, axis=1)
    emb_min = np.min(embeddings, axis=1)
    emb_range = emb_max - emb_min
    
    emb_stats = pd.DataFrame({
        'emb_mean': emb_mean,
        'emb_std': emb_std,
        'emb_max': emb_max,
        'emb_min': emb_min,
        'emb_range': emb_range
    })
    
    # 3. Normalized raw stats (8)
    X_stats = df[stats_cols].fillna(df[stats_cols].median())
    scaler = RobustScaler()
    X_stats_scaled = pd.DataFrame(
        scaler.fit_transform(X_stats),
        columns=stats_cols
    )
    
    # 4. Domain features (5)
    attack_score = (df['finishing'] + df['dribbling'] + df['sprint_speed']) / 3
    defense_score = (df['strength'] + df['stamina']) / 2
    midfield_score = (df['short_passing'] + df['stamina']) / 2
    speed_score = (df['acceleration'] + df['sprint_speed']) / 2
    technical_score = (df['dribbling'] + df['short_passing']) / 2
    
    domain_features = pd.DataFrame({
        'attack_score': attack_score,
        'defense_score': defense_score,
        'midfield_score': midfield_score,
        'speed_score': speed_score,
        'technical_score': technical_score
    })
    
    # Combine all features
    X = pd.concat([
        X_embedding.reset_index(drop=True),
        emb_stats.reset_index(drop=True),
        X_stats_scaled.reset_index(drop=True),
        domain_features.reset_index(drop=True)
    ], axis=1)
    
    logger.info(f"✅ Feature matrix: {X.shape}")
    logger.info(f"   - Embedding: {X_embedding.shape[1]} features")
    logger.info(f"   - Embedding stats: {emb_stats.shape[1]} features")
    logger.info(f"   - Normalized stats: {X_stats_scaled.shape[1]} features")
    logger.info(f"   - Domain features: {domain_features.shape[1]} features")
    
    return X, scaler


def train_model_with_embeddings(X, y, use_smote=True, test_size=0.2, random_state=42):
    """Train Random Forest with embeddings"""
    logger.info("\n🎯 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"📊 Train set: {X_train.shape[0]} samples")
    logger.info(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Apply SMOTE if enabled
    if use_smote:
        logger.info("⚖️  Applying SMOTE for class balance...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"   After SMOTE: {X_train.shape[0]} samples")
    
    # Train Random Forest
    logger.info("🌲 Training Random Forest...")
    
    best_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 3,
        'max_features': 'sqrt',
        'random_state': random_state,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n📈 Evaluating model...")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"✅ Train accuracy: {train_score:.4f}")
    logger.info(f"✅ Test accuracy: {test_score:.4f}")
    
    # Cross-validation
    logger.info("🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"✅ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, (X_train, X_test, y_train, y_test), (train_score, test_score, cv_scores.mean())


def save_model_artifacts(model, scaler, le, config, output_dir="../model"):
    """Save model and artifacts"""
    logger.info(f"\n💾 Saving model artifacts to {output_dir}...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/best_football_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✅ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = f"{output_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Saved scaler: {scaler_path}")
    
    # Save label encoder
    le_path = f"{output_dir}/label_encoder.pkl"
    joblib.dump(le, le_path)
    logger.info(f"✅ Saved label encoder: {le_path}")
    
    # Save config
    config_path = f"{output_dir}/model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Saved config: {config_path}")


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN MODEL WITH EMBEDDINGS")
    print("=" * 60 + "\n")
    
    # Configuration
    CSV_PATH = "../data/player_embeddings.csv"
    OUTPUT_DIR = "../model"
    
    stats_cols = [
        'age', 'acceleration', 'sprint_speed', 'dribbling',
        'short_passing', 'finishing', 'stamina', 'strength'
    ]
    
    try:
        # Load data
        df = load_embeddings_from_csv(CSV_PATH)
        
        # Build features
        X, scaler = build_features(df, stats_cols)
        
        # Encode labels
        logger.info("\n🏷️  Encoding labels...")
        le = LabelEncoder()
        y = le.fit_transform(df['primary_position'])
        logger.info(f"✅ Encoded {len(le.classes_)} position classes")
        
        # Train model
        model, splits, scores = train_model_with_embeddings(X, y, use_smote=True)
        train_score, test_score, cv_score = scores
        
        # Prepare config
        config = {
            "n_embedding_features": 64,
            "stats_cols": stats_cols,
            "use_smote": True,
            "model_type": "RandomForest",
            "accuracy": float(test_score),
            "cv_score": float(cv_score),
            "train_accuracy": float(train_score),
            "best_params": {
                "max_depth": 20,
                "max_features": "sqrt",
                "min_samples_split": 3,
                "n_estimators": 200
            },
            "feature_engineering": {
                "embedding_stats": True,
                "normalized_stats": True,
                "domain_features": True
            },
            "trained_at": datetime.now().isoformat(),
            "total_features": X.shape[1],
            "total_samples": len(df),
            "position_classes": le.classes_.tolist()
        }
        
        # Save artifacts
        save_model_artifacts(model, scaler, le, config, OUTPUT_DIR)
        
        print("\n" + "=" * 60)
        print("✅ STEP 4 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Model Performance:")
        print(f"  - Train accuracy: {train_score:.4f}")
        print(f"  - Test accuracy: {test_score:.4f}")
        print(f"  - CV score: {cv_score:.4f}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"  - Position classes: {len(le.classes_)}")
        print("\n🎯 Next: Run step5_predict_and_save.py or streamlit run app.py")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
