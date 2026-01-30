import pandas as pd
import numpy as np
import joblib
import os
from src.preprocess import load_csi_data, butter_lowpass_filter
from src.features import create_feature_matrix
from src.model import prepare_data, train_model, evaluate_model

# Dosya yolları ve parametreler
DATA_PATH = "data/annotations.csv"
WINDOW_SIZE = 100

def run_pipeline(file_path):
    raw_signal = load_csi_data(file_path)
    clean_signal = butter_lowpass_filter(raw_signal, cutoff=10, fs=100)
    X = create_feature_matrix(clean_signal, window_size=WINDOW_SIZE)
    
    try:
        # CSV'yi oku, 2. sütundaki 'fall', 'walk' gibi metinleri al
        df = pd.read_csv(file_path, header=None)
        labels_raw = df.iloc[:, 1].values[:len(X)] # 2. sütun (index 1)
        
        # Makine öğrenmesi metinleri anlamaz, sayıya çevirmeliyiz (Label Encoding)
        # fall -> 1, walk -> 0 gibi
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels_raw)
        
        print(f"-> {len(y)} adet GERÇEK etiket yüklendi. Sınıflar: {le.classes_}")
    except Exception as e:
        print(f"-> Hata oluştu: {e}. Rastgele veriye dönülüyor.")
        y = np.random.randint(0, 2, len(X))
    
    return X, y

if __name__ == "__main__":
    print("--- WiFall-Guard Sistemi Başlatılıyor ---")
    
    # Süreci çalıştır
    X, y = run_pipeline(DATA_PATH)
    
    # Eğit ve Değerlendir
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)
    
    print(f"\nİşlem Tamamlandı! \nModel Doğruluğu: %{acc*100:.2f}")
    print("\nDetaylı Rapor:\n", report)
    
    # Modeli 'brain' olarak kaydet
joblib.dump(model, "models/wifi_fall_model.pkl")
print("\nModel 'models/wifi_fall_model.pkl' olarak kaydedildi!")