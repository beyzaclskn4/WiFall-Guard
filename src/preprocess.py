import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def load_csi_data(file_path):
    # Veriyi başlık olmadan (veya varsa başlığıyla) oku
    df = pd.read_csv(file_path, header=None) 
    
    # Sadece İLK SÜTUNU (index 0) alıyoruz çünkü sayılar orada
    # errors='coerce' diyerek kazara araya karışan metinleri NaN yaparız
    signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    # NaN olanları (yani 'fall', 'benja' gibi kelimeleri) temizle
    signal = signal.dropna().values
    
    return signal
 
def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Alçak geçiren filtre uygulayarak gürültüyü temizler."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot_comparison(raw, clean, title="CSI Signal Analysis"):
    """Ham ve filtrelenmiş sinyalleri görselleştirir."""
    plt.figure(figsize=(12, 5))
    plt.plot(raw[:1000], label='Raw (Noisy)', alpha=0.4)
    plt.plot(clean[:1000], label='Clean (Filtered)', color='red', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.show()
    
    if __name__ == "__main__":
     DATA_PATH = "data/annotations.csv" 
    
    # 1. Yükle
    raw = load_csi_data(DATA_PATH)
    
    # 2. Temizle (100Hz örnekleme hızı, 10Hz kesme frekansı)
    clean = butter_lowpass_filter(raw, cutoff=10, fs=100)
    
    # 3. Görselleştir
    plot_comparison(raw, clean)
    print("Preprocessing completed successfully.")