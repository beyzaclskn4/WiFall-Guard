import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.preprocess import run_pipeline 
from src.features import create_feature_matrix 
from src.model import prepare_data, train_model, evaluate_model

DATA_PATH = "data/annotations.csv"
MODEL_DIR = "models"

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Sinyali temizle ve özellikleri çıkar
    clean_signal = run_pipeline(DATA_PATH)
    X = create_feature_matrix(clean_signal, window_size=100)
    
    # 'fall' ve 'walk' etiketlerini sayıya (0-1) çevir
    df = pd.read_csv(DATA_PATH, header=None)
    y_raw = df.iloc[:, 1].values[:len(X)] 
    y = LabelEncoder().fit_transform(y_raw)
    
    # Veriyi ölçeklendir (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim ve Test setlerine ayır, modeli eğit
    X_train, X_test, y_train, y_test = prepare_data(X_scaled, y)
    clf = train_model(X_train, y_train)
    
    # Sonuçları yazdır ve modeli dondur
    acc, report = evaluate_model(clf, X_test, y_test)
    print(f"Model Doğruluğu: %{acc*100:.2f}\n{report}")
    
    joblib.dump(clf, f"{MODEL_DIR}/wifi_fall_model.pkl")
    print(f"Başarıyla tamamlandı! Model {MODEL_DIR} içine kaydedildi.")