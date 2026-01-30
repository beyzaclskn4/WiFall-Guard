import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Model hazırlık ve veri bölme fonksiyonu
def prepare_data(feature_matrix, labels):
    """Veriyi eğitim ve test setlerine ayırır."""
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Random Forest modelini eğitir."""
    model = RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    return model

#neden randomforest kullnadık: Random Forest, yüksek doğruluk oranları sunan, aşırı öğrenmeye karşı dayanıklı ve çeşitli veri türlerinde iyi performans gösteren bir topluluk öğrenme algoritmasıdır. Ayrıca, hiperparametre ayarlamalarıyla esnekliği artırılabilir ve yorumlanabilir sonuçlar sağlar.
#wifi gibi gürültülü verilerde de iyi performans gösterir.
def evaluate_model(model, X_test, y_test):
    """Modelin başarısını ölçer ve raporlar."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

if __name__ == "__main__":
    # Test için sahte özellik matrisi ve etiketler (0: Normal, 1: Düşme)
    X_dummy = np.random.rand(100, 5) # 100 örnek, 5 özellik
    y_dummy = np.random.randint(0, 2, 100)
    
    # Süreçleri çalıştır
    X_train, X_test, y_train, y_test = prepare_data(X_dummy, y_dummy)
    clf = train_model(X_train, y_train)
    acc, rep = evaluate_model(clf, X_test, y_test)
    
    print(f"Model Doğruluğu: %{acc*100:.2f}")
    print("Sınıflandırma Raporu:\n", rep)