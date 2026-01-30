import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def load_csi_data(file_path):
    # Veriyi başlık olmadan oku
    df = pd.read_csv(file_path, header=None)
    # İlk sütundaki değerleri sayıya çevir, kelimeleri (fall, benja) NaN yap ve sil
    signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    return signal
 
def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Alçak geçiren filtre uygulayarak gürültüyü temizler."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def run_pipeline(file_path):
    """Sinyali yükler ve filtreden geçirir."""
    # 1. Ham sinyali yükle
    raw_signal = load_csi_data(file_path)
    # 2. 10 Hz cutoff frekansı ile temizle
    clean_signal = butter_lowpass_filter(raw_signal, cutoff=10, fs=100)
    return clean_signal