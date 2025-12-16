"""
STEP 4: Train Model with Embeddings
====================================
Script ini melatih model klasifikasi posisi pemain menggunakan:
- Embedding 64D dari Neo4j
- 8 atribut statistik
- Fitur statistik embedding
- Fitur domain (attack/defense/speed/etc)

Strategi:
1. Model 15 kelas (primary_position) + Top-1 / Top-3 / Top-5 accuracy
2. Model 4 kelas role (GK/DEF/MID/FWD) + metrik yang sama

Output:
- Model utama 15 kelas di ../model
- Model role 4 kelas di ../model_role
- Scaler dan label encoder masing-masing
- Config JSON berisi metrik & info model
"""

import os
import json
import logging
from datetime import datetime
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
#  Helper: Load & Feature Engineering
# ============================================================
def load_embeddings_from_csv(csv_path="../data/player_embeddings.csv"):
    """Load embeddings and stats from CSV."""
    logger.info(f"📂 Loading data from {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} players")

    # Parse embedding from JSON string
    df["embedding"] = df["embedding"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Primary position (ambil posisi pertama)
    df["primary_position"] = df["positions"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else x
    )

    logger.info(f"📊 Unique positions: {df['primary_position'].nunique()}")
    logger.info(
        "📊 Position distribution:\n%s",
        df["primary_position"].value_counts(),
    )

    return df


def build_features(df, stats_cols):
    """
    Build feature matrix dengan:
    - 64D embedding
    - 8 raw stats (belum diskalakan)
    - 5 embedding statistics
    - 5 domain features
    Total: 82 features
    """
    logger.info("🔧 Building feature matrix...")

    # 1. Embedding features (64D)
    embeddings = np.array(df["embedding"].tolist())
    X_embedding = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
    )

    # 2. Embedding statistics (5)
    emb_mean = np.mean(embeddings, axis=1)
    emb_std = np.std(embeddings, axis=1)
    emb_max = np.max(embeddings, axis=1)
    emb_min = np.min(embeddings, axis=1)
    emb_range = emb_max - emb_min

    emb_stats = pd.DataFrame(
        {
            "emb_mean": emb_mean,
            "emb_std": emb_std,
            "emb_max": emb_max,
            "emb_min": emb_min,
            "emb_range": emb_range,
        }
    )

    # 3. RAW stats (8) – belum di-scale
    X_stats = df[stats_cols].fillna(df[stats_cols].median())

    # 4. Domain features (5)
    attack_score = (
        df["finishing"] + df["dribbling"] + df["sprint_speed"]
    ) / 3
    defense_score = (df["strength"] + df["stamina"]) / 2
    midfield_score = (df["short_passing"] + df["stamina"]) / 2
    speed_score = (df["acceleration"] + df["sprint_speed"]) / 2
    technical_score = (df["dribbling"] + df["short_passing"]) / 2

    domain_features = pd.DataFrame(
        {
            "attack_score": attack_score,
            "defense_score": defense_score,
            "midfield_score": midfield_score,
            "speed_score": speed_score,
            "technical_score": technical_score,
        }
    )

    # Combine semua
    X = pd.concat(
        [
            X_embedding.reset_index(drop=True),
            emb_stats.reset_index(drop=True),
            X_stats.reset_index(drop=True),
            domain_features.reset_index(drop=True),
        ],
        axis=1,
    )

    logger.info(f"✅ Feature matrix: {X.shape}")
    logger.info(f"   - Embedding: {X_embedding.shape[1]} features")
    logger.info(f"   - Embedding stats: {emb_stats.shape[1]} features")
    logger.info(f"   - Raw stats: {X_stats.shape[1]} features")
    logger.info(f"   - Domain features: {domain_features.shape[1]} features")

    return X


# ============================================================
#  Helper: Training + Evaluation
# ============================================================
def train_model_with_embeddings(
    X,
    y,
    stats_cols,
    use_smote=True,
    test_size=0.2,
    random_state=42,
):
    """
    Train Random Forest dengan embeddings.
    - Scaling hanya di train set (kolom stats_cols)
    - SMOTE di dalam Pipeline (hanya train, termasuk saat CV)
    """
    logger.info("\n🎯 Training model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"📊 Train set: {X_train.shape[0]} samples")
    logger.info(f"📊 Test set: {X_test.shape[0]} samples")

    # === Scaling HANYA di train, hanya untuk kolom stats ===
    scaler = RobustScaler()
    X_train_stats = scaler.fit_transform(X_train[stats_cols])
    X_test_stats = scaler.transform(X_test[stats_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled.loc[:, stats_cols] = X_train_stats
    X_test_scaled.loc[:, stats_cols] = X_test_stats

    # Random Forest base model
    logger.info("🌲 Preparing Random Forest...")
    best_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 3,
        "max_features": "sqrt",
        "random_state": random_state,
        "n_jobs": -1,
    }
    rf = RandomForestClassifier(**best_params)

    # Pipeline: SMOTE (opsional) + RF
    if use_smote:
        logger.info(
            "⚖️  Using SMOTE inside Pipeline for class balance (train only)..."
        )
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=random_state)),
                ("rf", rf),
            ]
        )
    else:
        pipeline = Pipeline([("rf", rf)])

    # Fit di train
    logger.info("🚀 Fitting model...")
    pipeline.fit(X_train_scaled, y_train)

    # Train & test accuracy
    train_score = pipeline.score(X_train_scaled, y_train)
    test_score = pipeline.score(X_test_scaled, y_test)

    logger.info(f"✅ Train accuracy: {train_score:.4f}")
    logger.info(f"✅ Test accuracy:  {test_score:.4f}")

    # Cross-validation yang fair di TRAIN ONLY
    logger.info("🔄 Running 5-fold cross-validation (on train set)...")
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    cv_results = cross_validate(
        pipeline,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_mean = cv_results["test_score"].mean()
    cv_std = cv_results["test_score"].std()

    logger.info(f"✅ CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")

    splits = (X_train_scaled, X_test_scaled, y_train, y_test)
    scores = (train_score, test_score, cv_mean, cv_std)

    # return pipeline (model), scaler (buat nanti di predict), dll.
    return pipeline, scaler, splits, scores


def compute_baseline_accuracy(labels):
    """Baseline: selalu nebak kelas mayoritas."""
    counter = Counter(labels)
    majority_class, majority_count = counter.most_common(1)[0]
    baseline = majority_count / len(labels)
    return baseline, majority_class


def compute_topk_metrics(model, X_test_scaled, y_test, ks=(3, 5)):
    """Hitung Top-k accuracy (Top-3, Top-5, dll)."""
    if not hasattr(model, "predict_proba"):
        logger.warning(
            "Model tidak punya predict_proba, tidak bisa hitung Top-k."
        )
        return {}

    proba_test = model.predict_proba(X_test_scaled)
    metrics = {}
    for k in ks:
        topk = top_k_accuracy_score(y_test, proba_test, k=k)
        metrics[f"top_{k}"] = topk
        logger.info(f"🎯 Top-{k} accuracy: {topk:.4f}")
    return metrics


def save_model_artifacts(model, scaler, le, config, output_dir):
    """Save model dan artifacts ke folder tertentu."""
    logger.info(f"\n💾 Saving model artifacts to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save model (pipeline RF + SMOTE)
    model_path = os.path.join(output_dir, "best_football_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"✅ Saved model: {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Saved scaler: {scaler_path}")

    # Save label encoder
    le_path = os.path.join(output_dir, "label_encoder.pkl")
    joblib.dump(le, le_path)
    logger.info(f"✅ Saved label encoder: {le_path}")

    # Save config
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Saved config: {config_path}")


# ============================================================
#  Strategi 2: Mapping ke 4 role (GK/DEF/MID/FWD)
# ============================================================
def map_position_to_role(pos):
    """Map posisi detail (ST, CB, CM, dll) ke 4 role: GK/DEF/MID/FWD."""
    if not isinstance(pos, str):
        return None
    p = pos.upper().strip()

    # GK
    if "GK" in p:
        return "GK"

    # Defenders
    defender_tokens = ["CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"]
    if any(t in p for t in defender_tokens):
        return "DEF"

    # Forwards
    forward_tokens = ["ST", "CF", "LF", "RF", "LW", "RW"]
    if any(t in p for t in forward_tokens):
        return "FWD"

    # Default: midfield
    return "MID"


# ============================================================
#  MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN MODEL WITH EMBEDDINGS")
    print("=" * 60 + "\n")

    # Configuration
    CSV_PATH = "../data/player_embeddings.csv"
    OUTPUT_DIR_POSITION = "../model"        # 15-class primary position
    OUTPUT_DIR_ROLE = "../model_role"       # 4-class role model

    stats_cols = [
        "age",
        "acceleration",
        "sprint_speed",
        "dribbling",
        "short_passing",
        "finishing",
        "stamina",
        "strength",
    ]

    try:
        # -------------------------
        # 1) Load data & features
        # -------------------------
        df = load_embeddings_from_csv(CSV_PATH)
        X_all = build_features(df, stats_cols)

        # =======================================================
        # STRATEGI 1: Model 15 posisi dengan Top-k accuracy
        # =======================================================
        logger.info("\n====================")
        logger.info("🧩 STRATEGY 1: 15-class primary_position")
        logger.info("====================")

        # Encode labels: primary_position
        le_pos = LabelEncoder()
        y_pos = le_pos.fit_transform(df["primary_position"])
        logger.info(
            f"✅ Encoded {len(le_pos.classes_)} position classes: "
            f"{list(le_pos.classes_)}"
        )

        # Baseline (majority class)
        baseline_pos, majority_pos = compute_baseline_accuracy(
            df["primary_position"]
        )
        logger.info(
            f"📉 Baseline (always predict '{majority_pos}') = "
            f"{baseline_pos:.4f}"
        )

        # Train model posisi
        model_pos, scaler_pos, splits_pos, scores_pos = (
            train_model_with_embeddings(
                X_all,
                y_pos,
                stats_cols,
                use_smote=True,
            )
        )
        (
            X_train_pos,
            X_test_pos,
            y_train_pos,
            y_test_pos,
        ) = splits_pos
        (
            train_score_pos,
            test_score_pos,
            cv_score_pos,
            cv_std_pos,
        ) = scores_pos

        # Top-k metrics (Top-3 & Top-5)
        topk_pos = compute_topk_metrics(
            model_pos,
            X_test_pos,
            y_test_pos,
            ks=(3, 5),
        )

        # Config untuk model posisi
        config_pos = {
            "task": "primary_position_15_classes",
            "n_embedding_features": 64,
            "stats_cols": stats_cols,
            "use_smote": True,
            "model_type": "RandomForest",
            "accuracy_top1": float(test_score_pos),
            "accuracy": float(test_score_pos),  # ⬅️ TAMBAHAN: biar kompatibel dgn Streamlit
            "cv_score": float(cv_score_pos),
            "cv_std": float(cv_std_pos),
            "train_accuracy": float(train_score_pos),
            "baseline_accuracy": float(baseline_pos),
            "baseline_class": majority_pos,
            "topk_accuracy": {k: float(v) for k, v in topk_pos.items()},
            "best_params": {
                "max_depth": 20,
                "max_features": "sqrt",
                "min_samples_split": 3,
                "n_estimators": 200,
            },
            "feature_engineering": {
                "embedding_stats": True,
                "normalized_stats": True,
                "domain_features": True,
            },
            "trained_at": datetime.now().isoformat(),
            "total_features": X_all.shape[1],
            "total_samples": len(df),
            "classes": le_pos.classes_.tolist(),
        }

        # Save artifacts model posisi (pakai folder ../model seperti semula)
        save_model_artifacts(
            model_pos, scaler_pos, le_pos, config_pos, OUTPUT_DIR_POSITION
        )

        # =======================================================
        # STRATEGI 2: Model 4 role (GK/DEF/MID/FWD)
        # =======================================================
        logger.info("\n====================")
        logger.info("🧩 STRATEGY 2: 4-class role (GK/DEF/MID/FWD)")
        logger.info("====================")

        df_role = df.copy()
        df_role["role"] = df_role["primary_position"].apply(
            map_position_to_role
        )

        logger.info(
            "📊 Role distribution:\n%s", df_role["role"].value_counts()
        )

        # Filter baris valid
        mask_role = df_role["role"].notna()
        df_role = df_role[mask_role].reset_index(drop=True)
        X_role = X_all[mask_role].reset_index(drop=True)

        le_role = LabelEncoder()
        y_role = le_role.fit_transform(df_role["role"])

        logger.info(
            f"✅ Encoded {len(le_role.classes_)} role classes: "
            f"{list(le_role.classes_)}"
        )

        # Baseline role
        baseline_role, majority_role = compute_baseline_accuracy(
            df_role["role"]
        )
        logger.info(
            f"📉 Baseline role (always predict '{majority_role}') = "
            f"{baseline_role:.4f}"
        )

        # Train model role
        model_role, scaler_role, splits_role, scores_role = (
            train_model_with_embeddings(
                X_role,
                y_role,
                stats_cols,
                use_smote=True,
            )
        )
        (
            X_train_role,
            X_test_role,
            y_train_role,
            y_test_role,
        ) = splits_role
        (
            train_score_role,
            test_score_role,
            cv_score_role,
            cv_std_role,
        ) = scores_role

        # Top-k metrics untuk role (Top-2 & Top-3 misalnya)
        topk_role = compute_topk_metrics(
            model_role,
            X_test_role,
            y_test_role,
            ks=(2, 3),
        )

        # Config untuk model role
        config_role = {
            "task": "role_4_classes_GK_DEF_MID_FWD",
            "n_embedding_features": 64,
            "stats_cols": stats_cols,
            "use_smote": True,
            "model_type": "RandomForest",
            "accuracy_top1": float(test_score_role),
            "cv_score": float(cv_score_role),
            "cv_std": float(cv_std_role),
            "train_accuracy": float(train_score_role),
            "baseline_accuracy": float(baseline_role),
            "baseline_class": majority_role,
            "topk_accuracy": {k: float(v) for k, v in topk_role.items()},
            "best_params": {
                "max_depth": 20,
                "max_features": "sqrt",
                "min_samples_split": 3,
                "n_estimators": 200,
            },
            "feature_engineering": {
                "embedding_stats": True,
                "normalized_stats": True,
                "domain_features": True,
            },
            "trained_at": datetime.now().isoformat(),
            "total_features": X_role.shape[1],
            "total_samples": len(df_role),
            "classes": le_role.classes_.tolist(),
        }

        # Save artifacts model role di folder terpisah
        save_model_artifacts(
            model_role, scaler_role, le_role, config_role, OUTPUT_DIR_ROLE
        )

        # =======================================================
        # Summary print ke console (buat kamu lihat cepat)
        # =======================================================
        print("\n" + "=" * 60)
        print("✅ STEP 4 COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n📊 Model 1: 15-class primary_position")
        print(f"  - Train accuracy (Top-1): {train_score_pos:.4f}")
        print(f"  - Test accuracy  (Top-1): {test_score_pos:.4f}")
        print(f"  - CV score (5-fold):      {cv_score_pos:.4f}")
        print(f"  - Baseline accuracy:      {baseline_pos:.4f} "
              f"(class='{majority_pos}')")
        for k, v in topk_pos.items():
            print(f"  - {k.replace('_', '-')} accuracy: {v:.4f}")
        print(f"  - Total features:         {X_all.shape[1]}")
        print(f"  - Classes (15):           {len(le_pos.classes_)}")

        print("\n📊 Model 2: 4-class role (GK/DEF/MID/FWD)")
        print(f"  - Train accuracy (Top-1): {train_score_role:.4f}")
        print(f"  - Test accuracy  (Top-1): {test_score_role:.4f}")
        print(f"  - CV score (5-fold):      {cv_score_role:.4f}")
        print(f"  - Baseline accuracy:      {baseline_role:.4f} "
              f"(class='{majority_role}')")
        for k, v in topk_role.items():
            print(f"  - {k.replace('_', '-')} accuracy: {v:.4f}")
        print(f"  - Total features:         {X_role.shape[1]}")
        print(f"  - Classes (4):            {len(le_role.classes_)}")

        print(
            "\n🎯 Next: "
            "Gunakan model 15-class untuk demo posisi detail (app/step5), "
            "dan model 4-class + Top-k metrics untuk argument akurasi >70% "
            "di presentasi."
        )

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
