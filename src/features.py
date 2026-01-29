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

def create_feature_matrix(signal, window_size=100):
    """Sinyali pencerelere böler ve özellik matrisi oluşturur."""
    all_features = []
    
    # Sinyali belirlenen pencere boyutuyla tara
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i : i + window_size]
        features = extract_signal_features(window)
        all_features.append(features)
        
    return np.array(all_features)