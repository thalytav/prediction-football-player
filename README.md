# Prediksi Posisi Pemain Sepak Bola Menggunakan Graph Database dan Machine Learning

## Kelompok 10

- **Thalyta Vius Pramesti** (5025231055)
- **Winda Nafiqih Irawan** (5025231065)
- **Miskiyah** (5025231119)

## Latar Belakang

Perkembangan sepak bola modern tidak lepas dari pemanfaatan data untuk memahami karakteristik setiap pemain. Klub dan analis membutuhkan cara yang lebih terstruktur untuk melihat keterkaitan antara performa, atribut fisik, gaya bermain, serta peran yang dijalankan di lapangan.

Saat ini berbagai dataset pemain sudah tersedia secara terbuka. Namun, pemanfaatannya masih cenderung terbatas pada analisis konvensional. Padahal, data pemain memiliki pola hubungan yang alami dan saling terhubung, seperti relasi pemain dengan posisi, atribut teknik, maupun statistik performanya. Pola seperti ini sangat cocok dianalisis menggunakan pendekatan berbasis graph.

Dengan menggabungkan graph database dan model machine learning, analisis tersebut dapat dikembangkan lebih jauh. Pendekatan ini memungkinkan pemetaan atribut pemain secara lebih akurat sekaligus membantu memprediksi posisi yang paling sesuai berdasarkan struktur hubungan yang terbentuk dalam data.

## Rumusan Masalah

1. Bagaimana memodelkan hubungan natural antar entitas seperti pemain, atribut, dan posisi dalam graph database?
2. Bagaimana memanfaatkan dataset untuk memprediksi posisi pemain secara otomatis?

## Tujuan

1. Membuat struktur graph yang mewakili hubungan pemain dan atributnya.
2. Menerapkan model machine learning untuk memprediksi posisi pemain berdasarkan atribut.

## Project Overview

Proyek ini bertujuan untuk memprediksi posisi pemain sepak bola berdasarkan metrik performa mereka. Dengan memanfaatkan kombinasi graph database dan machine learning, proyek ini memberikan pendekatan yang lebih terstruktur dan akurat dalam analisis data pemain.

## Fitur Utama

- **Pembersihan Data**: Memproses data mentah pemain sepak bola untuk pelatihan model.
- **Pelatihan Model**: Melatih dan menyimpan model Random Forest dengan hyperparameter yang dioptimalkan.
- **Aplikasi Interaktif**: Memungkinkan prediksi posisi pemain melalui antarmuka Streamlit yang ramah pengguna.
- **Real-Time Neo4j Integration**: Sistem otomatis menyimpan data pemain baru ke Neo4j setelah prediksi, termasuk:
  - Node pemain dengan semua atribut teknis
  - 64-dimensional graph embedding
  - Domain scores (attack, defense, midfield, speed, technical)
  - Relationship PLAYS_AS ke posisi yang diprediksi
- **Graph Explorer**: Visualisasi network graph interaktif dari hubungan pemain berdasarkan similarity embeddings dari Neo4j dengan 3 mode:
  - Position Clusters: Pemain dikelompokkan berdasarkan posisi
  - Player Similarity: Graph berdasarkan cosine similarity embedding
  - Position Hierarchy: Struktur hierarki posisi pemain
- **Dataset Analysis**: Analisis statistik lengkap dengan visualisasi distribusi posisi, usia, korelasi atribut, dan stats per posisi.
- **Visualisasi**: Menyediakan grafik penting seperti feature importance, perbandingan model, dan interactive network graphs.

## Struktur File

```
prediction-football-player/
├── script/
│   ├── app.py                    # Aplikasi Streamlit untuk prediksi
│   ├── train_model.py            # Skrip pelatihan model
│   ├── clean_data.py             # Skrip preprocessing data
│   ├── neo4j_connector.py        # Modul koneksi Neo4j
│   └── analyze_data.py           # Skrip analisis data
├── model/
│   ├── best_football_model.pkl   # Model terlatih
│   ├── scaler.pkl                # Scaler untuk normalisasi
│   ├── label_encoder.pkl         # Encoder untuk label posisi
│   └── model_config.json         # Konfigurasi model
├── data/
│   ├── cleaned_football_data.csv # Dataset terproses
│   └── neo4j_query_table_data_2025-11-27.csv  # Data mentah
├── .env                          # Konfigurasi Neo4j (buat dari .env.example)
├── .env.example                  # Template konfigurasi
├── requirements.txt              # Dependensi Python
├── .gitignore                    # File yang diabaikan git
└── README.md                     # Dokumentasi ini
```

## Instalasi

1. Clone repositori:
   ```bash
   git clone https://github.com/thalytav/prediction-football-player.git
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd prediction-football-player
   ```
3. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Untuk Windows
   source .venv/bin/activate  # Untuk macOS/Linux
   ```
4. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

5. Konfigurasi Neo4j (untuk fitur real-time integration):
   ```bash
   # Copy file .env.example ke .env
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   
   # Edit .env dan isi dengan kredensial Neo4j Anda
   ```
   
   Isi file `.env`:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   ```

## Penggunaan

### Melatih Model

1. Pastikan dataset telah diproses dan tersedia sebagai `cleaned_football_data.csv`.
2. Jalankan skrip pelatihan:
   ```bash
   python train_model.py
   ```

### Menjalankan Aplikasi

1. Pastikan Neo4j database berjalan (jika ingin menggunakan fitur real-time integration):
   ```bash
   # Jika menggunakan Neo4j Desktop, start database dari aplikasi
   # Jika menggunakan Docker:
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:latest
   ```

2. Jalankan aplikasi Streamlit:
   ```bash
   cd script
   streamlit run app.py
   ```

3. Buka aplikasi di browser pada `http://localhost:8501`.

4. Untuk menggunakan fitur Neo4j:
   - Di sidebar, klik "Configure Neo4j Connection"
   - Centang "Enable Neo4j Integration"
   - Isi kredensial Neo4j atau gunakan yang sudah dikonfigurasi di `.env`
   - Input nama pemain dan atribut teknis
   - Klik "Prediksi Posisi Sekarang"
   - Sistem akan otomatis menyimpan data pemain ke Neo4j setelah prediksi

## Dependensi

- Python 3.8+
- Libraries: 
  - Streamlit (UI framework)
  - scikit-learn (machine learning)
  - pandas, numpy (data processing)
  - joblib (model serialization)
  - networkx, pyvis (graph visualization)
  - matplotlib, seaborn (plotting)
  - neo4j (database driver)
  - python-dotenv (environment configuration)

## Alur Kerja Real-Time Integration

1. **User Input**: Pengguna memasukkan nama dan atribut teknis pemain baru melalui UI Streamlit
2. **Feature Engineering**: Sistem mengekstrak 82 features dari input:
   - 64 dimensi embedding (dari pemain terdekat di dataset)
   - 8 atribut teknis yang dinormalisasi
   - 5 embedding statistics
   - 5 domain-specific scores
3. **Prediction**: Model Random Forest memprediksi posisi optimal
4. **Neo4j Insertion**: Jika enabled, sistem otomatis:
   - Membuat/update node Player dengan semua atribut
   - Menyimpan 64-dimensional embedding
   - Membuat relationship PLAYS_AS ke node Position
   - Mencatat timestamp created_at dan updated_at
5. **Visualization**: Hasil prediksi ditampilkan dengan confidence scores dan logic reasoning

## Fitur Graph Explorer

Graph Explorer memungkinkan eksplorasi visual dari dataset pemain dalam bentuk network graph interaktif:

### Mode Visualisasi:

1. **Position Clusters**: Menampilkan cluster pemain berdasarkan posisi mereka
2. **Player Similarity**: Graph berdasarkan cosine similarity dari embeddings Neo4j
3. **Position Hierarchy**: Struktur hierarki yang mengelompokkan posisi ke dalam kategori (Attack, Midfield, Defense, Goalkeeper)

### Filter Options:

- Filter berdasarkan posisi spesifik
- Atur jumlah pemain yang ditampilkan (10-100)
- Atur threshold similarity untuk edges (0.5-0.95)

### Interaksi:

- Hover pada node untuk melihat detail pemain
- Drag node untuk reposisi
- Zoom in/out untuk eksplorasi detail
- Click node untuk highlight connections
