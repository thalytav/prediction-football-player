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
- **Visualisasi**: Menyediakan grafik penting seperti feature importance dan perbandingan model.

## Struktur File
- `app.py`: Aplikasi Streamlit untuk prediksi.
- `train_model.py`: Skrip untuk melatih dan menyimpan model.
- `clean_data.py`: Skrip untuk preprocessing data.
- `requirements.txt`: Daftar dependensi Python.
- `model_config.json`: File konfigurasi model.
- `best_football_model.pkl`: File model yang telah dilatih.
- `scaler.pkl`, `label_encoder.pkl`: Artefak preprocessing.

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

## Penggunaan
### Melatih Model
1. Pastikan dataset telah diproses dan tersedia sebagai `cleaned_football_data.csv`.
2. Jalankan skrip pelatihan:
   ```bash
   python train_model.py
   ```

### Menjalankan Aplikasi
1. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Buka aplikasi di browser pada `http://localhost:8501`.

## Dependensi
- Python 3.8+
- Library: Streamlit, scikit-learn, pandas, numpy, joblib

