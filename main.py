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