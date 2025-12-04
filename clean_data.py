import pandas as pd
import unidecode
import re

# 1. Load Data Mentah
# Kita coba baca pake encoding 'utf-8'. Kalau error, dia bakal coba 'latin-1' otomatis.
file_path = 'neo4j_query_table_data_2025-11-27.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("Gagal baca UTF-8, mencoba Latin-1...")
    df = pd.read_csv(file_path, encoding='latin-1')

print(f"Data awal dimuat: {len(df)} baris.")

# 2. Fungsi Pembersih Ajaib
def clean_text(text):
    if not isinstance(text, str):
        return text
    
    # Langkah A: Unidecode (Ubah aksen jadi huruf biasa)
    # Contoh: "Andrés" -> "Andres", "João" -> "Joao"
    text = unidecode.unidecode(text)
    
    # Langkah B: Regex Cleaning (Hapus simbol aneh)
    # Cuma bolehin Huruf (a-z), Spasi, Titik (.), Koma (,), dan Strip (-)
    # Simbol kotak-kotak atau emoticon bakal ilang
    text = re.sub(r'[^a-zA-Z0-9\s\.,\-]', '', text)
    
    # Hapus spasi ganda kalau ada
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Eksekusi Pembersihan
print("Sedang membersihkan nama pemain...")
df['full_name'] = df['full_name'].apply(clean_text)

# (Opsional) Bersihin kolom posisi juga biar gak ada spasi aneh
df['positions'] = df['positions'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

# 4. Cek Hasil (Before vs After)
# Kita tampilin sampel yang biasanya ribet
sample_names = ["Messi", "Ronaldo", "Neymar"] # Contoh aja
print("\n--- Sampel Data Bersih ---")
print(df[['full_name', 'positions']].head(5))

# 5. Simpan ke CSV Baru (File ini yang nanti dipake buat App & Training)
new_file_name = 'cleaned_football_data.csv'
df.to_csv(new_file_name, index=False, encoding='utf-8')

print(f"\n SUKSES! Data bersih disimpan sebagai '{new_file_name}'")

print("Pake file yang BARU ini untuk training model dan app.py ya!")
