import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pickle
import os
from scipy.sparse import save_npz

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Fungsi Pembersihan Teks ---
def clean_text(text, stop_words):
    """
    Membersihkan teks mentah:
    1. Hapus tag HTML & entitas
    2. Ubah ke huruf kecil
    3. Hapus tanda baca dan angka
    4. Hapus stopwords
    """
    # 1. Hapus tag HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Hapus entitas HTML seperti &amp;
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
    
    # 2. Ubah ke huruf kecil
    text = text.lower()
    
    # 3. Hapus tanda baca dan angka (simpan hanya huruf dan spasi)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Hapus stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Gabungkan kembali
    return " ".join(cleaned_words)

# --- Fungsi Utama untuk Menjalankan Pipeline ---
def main():
    """
    Fungsi utama untuk menjalankan pipeline preprocessing.
    """
    print("Memulai pipeline preprocessing...")

    # --- 1. Definisi Path ---
    # Path relatif dari lokasi skrip (preprocessing/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "../dataset-coursera_raw/courses_en.csv")
    
    # Sesuai Kriteria 1, simpan di folder 'namadataset_preprocessing'
    OUTPUT_DIR = os.path.join(BASE_DIR, "../preprocessing/dataset-coursera_preprocessing")
    
    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Download Stopwords ---
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # --- 3. Load Data ---
    print(f"Memuat data dari {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {INPUT_PATH}")
        return

    # --- 4. Penanganan Missing Values & Duplikat ---
    print("Menangani missing values dan duplikat...")
    df['what_you_learn'] = df['what_you_learn'].fillna("")
    df.drop_duplicates(subset='url', keep='first', inplace=True)

    # --- 5. Pembuatan Fitur Teks Gabungan ---
    df['combined_text'] = df['name'] + " " + \
                          df['what_you_learn'] + " " + \
                          df['skills'] + " " + \
                          df['content']

    # --- 6. Pembersihan Teks ---
    print("Membersihkan teks (ini mungkin perlu waktu)...")
    df['cleaned_text'] = df['combined_text'].apply(lambda text: clean_text(text, stop_words))
    print("Pembersihan teks selesai.")

    # --- 7. Encoding Target ---
    print("Melakukan encoding pada target (category)...")
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # --- 8. Train-Test Split ---
    print("Melakukan train-test split (80/20)...")
    X = df['cleaned_text']
    y = df['category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)

    # --- 9. Vektorisasi TF-IDF ---
    print("Menerapkan TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"Bentuk X_train_tfidf: {X_train_tfidf.shape}")

    # --- 10. Simpan Artefak Preprocessing ---
    print(f"Menyimpan artefak ke {OUTPUT_DIR}...")
    
    # Simpan data TF-IDF (format .npz untuk sparse matrix)
    save_npz(os.path.join(OUTPUT_DIR, "X_train.npz"), X_train_tfidf)
    save_npz(os.path.join(OUTPUT_DIR, "X_test.npz"), X_test_tfidf)
    
    # Simpan y (target) menggunakan pickle
    with open(os.path.join(OUTPUT_DIR, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(OUTPUT_DIR, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)
        
    # Simpan "perkakas" (vectorizer dan encoder)
    with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
        
    print("="*30)
    print("PIPELINE PREPROCESSING SELESAI")
    print(f"Data yang diproses dan artefak disimpan di: {OUTPUT_DIR}")
    print("="*30)

# --- Entry point untuk skrip ---
if __name__ == "__main__":
    main()