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

def create_feature_matrix(signal, window_size=100, step_size=20): # Step eklendi
    all_features = []
    # Sinyali 100'er atlayarak değil, 20'şer kaydırarak oku (Daha fazla veri!)
    for i in range(0, len(signal) - window_size, step_size):
        window = signal[i : i + window_size]
        features = extract_signal_features(window)
        all_features.append(features)
    return np.array(all_features)

if __name__ == "__main__":
    # Test amaçlı dummy (sahte) sinyal üretimi
    sample_signal = np.random.normal(0, 1, 1000)
    
    feature_matrix = create_feature_matrix(sample_signal, window_size=100)
    
    print(f"Sinyal işlendi. Matris boyutu: {feature_matrix.shape}")
    print("Özellik çıkarımı başarıyla tamamlandı.")