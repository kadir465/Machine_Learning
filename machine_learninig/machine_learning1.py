# import işlemleri yapılmalı
# 1-veri seti işlemleri yapılmalı
# 2-makine öğrenim modeli seçilmeli
# 3-modelin train edilmesi
# 4- sonuçların değerlendirmesi

from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Machin_lessons1:
    def __init__(self):
        # --- 1. Veri Setini Yükleme ---
        # sklearn.datasets içindeki göğüs kanseri (breast cancer) veri setini yüklüyoruz.
        cancer = load_breast_cancer()

        # --- 2. Pandas DataFrame ile Veri İnceleme ---
        # Daha rahat analiz edebilmek için veriyi DataFrame'e dönüştürüyoruz.
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df["target"] = cancer.target   # 0 = malign, 1 = benign

        # Bağımsız değişkenler (X) -> tüm özellikler
        # Bağımlı değişken (y) -> hedef etiket (0 veya 1)
        X = cancer.data
        y = cancer.target

        # --- 3. Eğitim ve Test Setine Ayırma ---
        # Veri setini %70 eğitim, %30 test olacak şekilde ayırıyoruz.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=45
        )

        # --- 4. Özellik Ölçekleme ---
        # KNN (K-En Yakın Komşu) algoritması, uzaklıklara göre çalışır.
        # Eğer özellikler farklı ölçeklerde olursa (ör. cm, gram, sayı),
        # !model yanlış öğrenebilir. Bu yüzden StandardScaler ile her özelliği
        # ortalaması 0, standart sapması 1 olacak şekilde ölçekliyoruz.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Eğitim verisini öğren + dönüştür
        X_test = scaler.transform(X_test)        # Test verisini aynı dönüşümle uygula

        # --- 5. KNN Modelini Oluşturma ---
        # Varsayılan parametrelerle (n_neighbors=5) KNN sınıflandırıcısını kuruyoruz.
        knn = KNeighborsClassifier(n_neighbors=3)

        # Modeli eğitiyoruz -> eğitim verisini kullanarak sınıfları öğreniyor.
        knn.fit(X_train, y_train)

        # --- 6. Tahmin Yapma ---
        # Test setindeki X_test verileri için tahmin yapıyoruz.
        y_pred = knn.predict(X_test)

        # --- 7. Model Performansı ---
        # accuracy_score ile tahminlerin doğruluk oranını ölçüyoruz.
        accuracy = accuracy_score(y_test, y_pred)
        print("Doğruluk : ", accuracy)
        
# --- Confusion Matrix ---
        # Confusion matrix, gerçek etiketlerle (y_test) modelin tahminlerini (y_pred) karşılaştırır.
        # Matriste satırlar -> gerçek sınıflar, sütunlar -> modelin tahmin ettiği sınıflardır.
        # load_breast_cancer veri setinde:
        #   0 = malign (kötü huylu)
        #   1 = benign (iyi huylu)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)



Machin_lessons1()
