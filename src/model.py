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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

#neden randomforest kullnadık: Random Forest, yüksek doğruluk oranları sunan, aşırı öğrenmeye karşı dayanıklı ve çeşitli veri türlerinde iyi performans gösteren bir topluluk öğrenme algoritmasıdır. Ayrıca, hiperparametre ayarlamalarıyla esnekliği artırılabilir ve yorumlanabilir sonuçlar sağlar.
#wifi gibi gürültülü verilerde de iyi performans gösterir.
