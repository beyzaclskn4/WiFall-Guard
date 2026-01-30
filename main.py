import pandas as pd
import numpy as np
import joblib
from src.preprocess import load_csi_data, butter_lowpass_filter
from src.features import create_feature_matrix
from src.model import prepare_data, train_model, evaluate_model

# Dosya yolları ve parametreler
DATA_PATH = "data/annotations.csv"
WINDOW_SIZE = 100

def run_pipeline(file_path):
    # 1. Veriyi yükle ve temizle
    raw_signal = load_csi_data(file_path)
    clean_signal = butter_lowpass_filter(raw_signal, cutoff=10, fs=100)
    
    # 2. Özellikleri çıkar
    X = create_feature_matrix(clean_signal, window_size=WINDOW_SIZE)
    
    # 3. GERÇEK ETİKETLERİ YÜKLE (CSV'den)
    # Varsayıyoruz ki CSV'de 'label' isminde bir sütun var (0: Normal, 1: Düşme)
    try:
        df_labels = pd.read_csv(file_path)
        # Sinyal pencerelendiği için etiket sayısını X'e uydurmamız gerekir
        y = df_labels['label'].values[:len(X)] 
    except:
        print("Uyarı: Gerçek etiketler yüklenemedi, test modunda devam ediliyor.")
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