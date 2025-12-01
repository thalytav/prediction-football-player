import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

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
# -----------------------------------------------

# Encode label
le = LabelEncoder()
y = le.fit_transform(df['primary_position'])

# Features (Embedding)
X = pd.DataFrame(df['embedding'].tolist())

# Split Data (Ini yang bikin urutan jadi acak/shuffled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. MODEL COMPARISON (BUKTI KUAT) ---
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("=== HASIL KOMPARASI MODEL (5-Fold CV) ===")
results = []
names = []
for name, model in models.items():
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: Akurasi {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.ylabel('Accuracy')
plt.savefig('model_comparison_plot.png') 
print("\nPlot tersimpan sebagai 'model_comparison_plot.png'")

# --- 3. FINAL MODEL & REPORT ---
print("\n=== TRAINING FINAL MODEL (RANDOM FOREST) ===")
final_model = models['Random Forest']
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))