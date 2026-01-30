import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.preprocess import run_pipeline 
from src.model import prepare_data, train_model, evaluate_model

# Ayarlar
DATA_PATH = "data/annotations.csv"
MODEL_DIR = "models"

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Veriyi Al
    X, y = run_pipeline(DATA_PATH)
    
    # 2. Sayıları "Eşitle" (Scaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Eğit ve Test Et
    X_train, X_test, y_train, y_test = prepare_data(X_scaled, y)
    clf = train_model(X_train, y_train)
    acc, report = evaluate_model(clf, X_test, y_test)
    
    # 4. Sonucu Yazdır
    print(f"Model Doğruluğu: %{acc*100:.2f}")
    print(report)
    
    # 5. Kaydet (Beyni sakla)
    joblib.dump(clf, f"{MODEL_DIR}/wifi_fall_model.pkl")