import numpy as np

def extract_signal_features(signal_segment):
    """Sinyal diliminden istatistiksel öznitelikler çıkarır."""
    features = []
    # Zaman boyutu özellikleri
    features.append(np.mean(signal_segment))    # Ortalama
    features.append(np.std(signal_segment))     # Standart Sapma
    features.append(np.var(signal_segment))     # Varyans
    features.append(np.max(signal_segment) - np.min(signal_segment)) # Genlik farkı
    
    # Enerji (Düşme tespiti için en kritik veri)
    energy = np.sum(np.square(signal_segment))
    features.append(energy)
    
    return np.array(features)