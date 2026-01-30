import numpy as np
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
    
    # 2. Özellikleri çıkar (Feature Matrix)
    X = create_feature_matrix(clean_signal, window_size=WINDOW_SIZE)
    
    # 3. Etiketleri oluştur (Şimdilik test için rastgele)
    y = np.random.randint(0, 2, len(X))
    
    return X, y
