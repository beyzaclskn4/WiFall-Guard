import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def load_csi_data(file_path):
    """CSV formatındaki CSI verisini yükler."""
    df = pd.read_csv(file_path)
    # İlk sütun zaman/indeks, ikinci sütun genlik varsayılmıştır
    raw_signal = df.iloc[:, 1].values 
    return raw_signal
 
def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Alçak geçiren filtre uygulayarak gürültüyü temizler."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
