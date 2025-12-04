import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib
import pickle
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

# ==================================================
# CONFIGURATION
# ==================================================
USE_GPU = True       # Set True to use PyTorch GPU acceleration
FAST_MODE = False     # Set True for faster training (3 folds instead of 5)
# ==================================================

# Check GPU availability
GPU_AVAILABLE = False
if USE_GPU:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            device = torch.device('cuda')
            print(f" GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Using PyTorch {torch.__version__}")
        else:
            print("  PyTorch installed but no CUDA GPU detected")
            device = torch.device('cpu')
    except ImportError:
        print("  GPU mode enabled but PyTorch not installed")
        print("     Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n     Or for RTX 3060 (CUDA 12.x):")
        print("    conda install -c rapidsai -c conda-forge -c nvidia rapids=24.12 python=3.11 cuda-version=12.0")
        GPU_AVAILABLE = False
        USE_GPU = False
        device = torch.device('cpu')
else:
    GPU_AVAILABLE = False
    print(" Running in CPU mode (set USE_GPU=True for GPU acceleration)")
    device = None

if FAST_MODE:
    print(" FAST MODE enabled (3-fold CV, reduced param grid)")
else:
    print(" FULL MODE (5-fold CV, extensive param grid)")

print("="*60)
start_time = time.time()

# --- 1. PREPARATION ---
print("1. Loading Data...")
df = pd.read_csv('cleaned_football_data.csv')

# Parsing kolom embedding
df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Parsing Target Position (Ambil yang pertama saja)
df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

# --- CEK VALIDASI LOGIKA (BUAT MEMASTIKAN CF) ---
print("\n[Validasi Data Awal]")
print(f"Pemain Pertama (CSV): {df.iloc[0]['full_name']}")
print(f"Posisi Asli (CSV): {df.iloc[0]['positions']}")
print(f"Posisi Utama (Logic): {df.iloc[0]['primary_position']}")
print("(Harusnya CF kalau datanya Messi 'CF,RW,ST')\n")

# Check class distribution
print(f"[Distribusi Kelas]")
print(df['primary_position'].value_counts())
print()

# --- 2. ADVANCED FEATURE ENGINEERING ---
print("2. Feature Engineering...")

# A. Embedding features
X_embedding = pd.DataFrame(df['embedding'].tolist())

# B. Statistical features dari embedding (tambahan info agregat)
df['emb_mean'] = df['embedding'].apply(np.mean)
df['emb_std'] = df['embedding'].apply(np.std)
df['emb_max'] = df['embedding'].apply(np.max)
df['emb_min'] = df['embedding'].apply(np.min)
df['emb_range'] = df['emb_max'] - df['emb_min']

# C. Raw statistics (normalized)
stats_cols = ['age', 'acceleration', 'sprint_speed', 'dribbling', 
              'short_passing', 'finishing', 'stamina', 'strength']
X_stats = df[stats_cols].fillna(df[stats_cols].median())

# Normalize stats to 0-1 range
scaler_stats = RobustScaler()
X_stats_scaled = pd.DataFrame(
    scaler_stats.fit_transform(X_stats),
    columns=stats_cols
)

# D. Domain-specific features
df['attack_score'] = (df['finishing'] + df['dribbling'] + df['sprint_speed']) / 3
df['defense_score'] = (df['strength'] + df['stamina']) / 2
df['midfield_score'] = (df['short_passing'] + df['stamina']) / 2
df['speed_score'] = (df['acceleration'] + df['sprint_speed']) / 2
df['technical_score'] = (df['dribbling'] + df['short_passing']) / 2

# E. Aggregate features
emb_stats = df[['emb_mean', 'emb_std', 'emb_max', 'emb_min', 'emb_range']]
domain_features = df[['attack_score', 'defense_score', 'midfield_score', 
                       'speed_score', 'technical_score']]

# F. Combine ALL features
X = pd.concat([
    X_embedding,           # Original embeddings (64 dims)
    emb_stats.reset_index(drop=True),      # Embedding statistics (5 dims)
    X_stats_scaled.reset_index(drop=True), # Normalized raw stats (8 dims)
    domain_features.reset_index(drop=True) # Domain features (5 dims)
], axis=1)

# Fix column names to all be strings
X.columns = X.columns.astype(str)

print(f"Total features: {X.shape[1]} (embedding: {X_embedding.shape[1]}, stats: {len(stats_cols)}, engineered: {emb_stats.shape[1] + domain_features.shape[1]})")

# --- 3. ENCODE LABELS ---
# Encode label
le = LabelEncoder()
y = le.fit_transform(df['primary_position'])


# --- 4. HANDLE CLASS IMBALANCE ---
print("\n3. Handling Class Imbalance with SMOTE...")
# Check if we need SMOTE
class_counts = pd.Series(y).value_counts()
min_class_count = class_counts.min()

# Only apply SMOTE if smallest class has at least 6 samples
if min_class_count >= 6:
    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original dataset: {X.shape[0]} samples")
    print(f"After SMOTE: {X_resampled.shape[0]} samples")
else:
    X_resampled, y_resampled = X, y
    print(f"Skipping SMOTE (smallest class has {min_class_count} samples)")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled
)

# --- 5. MODEL COMPARISON WITH OPTIMIZED CONFIGS ---
cv_folds = 3 if FAST_MODE else 5
print(f"\n=== HASIL KOMPARASI MODEL ({cv_folds}-Fold CV) ===")

# Always use sklearn (GPU doesn't help much for tree-based models in sklearn)
# But we optimize with n_jobs=-1 to use all CPU cores
print(" Using optimized CPU-based models (scikit-learn with parallel processing)")
print(f"   Available CPU cores: {joblib.cpu_count()}")
if GPU_AVAILABLE:
    print(f"   GPU: {torch.cuda.get_device_name(0)} (for neural networks if needed)")

n_estimators = 100 if FAST_MODE else 200
models = {
    'Random Forest (Tuned)': RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    ),
    'Random Forest (Simple)': RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
}

if not FAST_MODE:
    # Add more models only in full mode
    models.update({
        'SVM (RBF)': SVC(
            kernel='rbf', 
            C=10,
            gamma='scale',
            probability=True, 
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski'
        )
    })

results = []
names = []
for name, model in models.items():
    print(f"Training {name}...")
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_resampled, y_resampled, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"   {name}: Akurasi {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

# Visualisasi Perbandingan
plt.figure(figsize=(12, 6))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison (After Feature Engineering & SMOTE)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_plot.png', dpi=150)
print("\n Plot tersimpan sebagai 'model_comparison_plot.png'")

# --- 6. HYPERPARAMETER TUNING UNTUK BEST MODEL ---
print("\n=== HYPERPARAMETER TUNING (Random Forest) ===")

if FAST_MODE:
    # FAST MODE: Reduced parameter grid
    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [15, 20],
        'min_samples_split': [3, 5],
        'max_features': ['sqrt']
    }
    tuning_cv_folds = 3
    print(" FAST MODE: Using reduced parameter grid (3-fold CV)")
else:
    # FULL MODE: Complete parameter grid
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [15, 20, 25],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2']
    }
    tuning_cv_folds = 5
    print(" FULL MODE: Using complete parameter grid (5-fold CV)")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)
grid_search = GridSearchCV(
    rf_base, 
    param_grid, 
    cv=tuning_cv_folds, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  # Show progress
)
print(" Searching best parameters...")
grid_search.fit(X_train, y_train)

print(f"\n Best parameters: {grid_search.best_params_}")
print(f" Best CV score: {grid_search.best_score_:.4f}")

# --- 7. FINAL MODEL EVALUATION ---
print("\n=== TRAINING FINAL MODEL (BEST RANDOM FOREST) ===")
final_model = grid_search.best_estimator_

y_pred = final_model.predict(X_test)
test_score = final_model.score(X_test, y_test)

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature Importance (Top 15)
feature_names = (
    [f'emb_{i}' for i in range(X_embedding.shape[1])] +
    ['emb_mean', 'emb_std', 'emb_max', 'emb_min', 'emb_range'] +
    stats_cols +
    ['attack_score', 'defense_score', 'midfield_score', 'speed_score', 'technical_score']
)

importances = final_model.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)

# Calculate training time
training_time = time.time() - start_time
mins, secs = divmod(training_time, 60)

print(f"\nFinal Test Accuracy: {test_score:.4f}")
print(f"Total Training Time: {int(mins)}m {int(secs)}s")
print(f"Mode: {'FAST' if FAST_MODE else 'FULL'} | {'GPU' if USE_GPU and GPU_AVAILABLE else 'CPU'}")

# --- 8. SAVE MODEL & ARTIFACTS ---
print("\n=== SAVING MODEL & ARTIFACTS ===")

# Save model
model_filename = 'best_football_model.pkl'
joblib.dump(final_model, model_filename)

# Save label encoder
le_filename = 'label_encoder.pkl'
joblib.dump(le, le_filename)

# Save scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler_stats, scaler_filename)

# Save feature configuration for app.py
feature_config = {
    'n_embedding_features': X_embedding.shape[1],
    'stats_cols': stats_cols,
    'use_smote': True,
    'model_type': 'RandomForest',
    'accuracy': float(test_score),
    'cv_score': float(grid_search.best_score_),
    'best_params': grid_search.best_params_,
    'feature_engineering': {
        'embedding_stats': True,
        'normalized_stats': True,
        'domain_features': True
    }
}

config_filename = 'model_config.json'
with open(config_filename, 'w') as f:

    json.dump(feature_config, f, indent=2)
