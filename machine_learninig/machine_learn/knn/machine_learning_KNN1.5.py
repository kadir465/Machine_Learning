from sklearn.datasets import load_breast_cancer   # sklearn kütüphanesinden göğüs kanseri veri setini yüklüyoruz
from sklearn.model_selection import train_test_split  # veriyi eğitim ve test olarak ayırmak için
from sklearn.neighbors import KNeighborsClassifier    # KNN algoritması için sınıf
import pandas as pd   # veri işleme kütüphanesi
import matplotlib.pyplot as plt   # grafik çizdirmek için
from sklearn.metrics import accuracy_score   # doğruluk ölçümü için

class Machine_learn1_5():
    def __init__(self):
        # Göğüs kanseri veri setini yüklüyoruz
        cancer = load_breast_cancer()
        
        # Veri setini pandas DataFrame'e dönüştürüp, hedef (target) kolonunu ekliyoruz
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df["target"] = cancer.target

        # Özellikler (X) ve hedef değişkeni (y) ayırıyoruz
        X = cancer.data
        y = cancer.target

        # Doğruluk değerlerini ve k değerlerini saklamak için boş listeler oluşturuyoruz
        accuracy_value = []
        k_value = []

        # Veriyi %70 eğitim, %30 test olacak şekilde ayırıyoruz
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
        
        # KNN algoritması için k değerlerini 1’den 20’ye kadar deniyoruz
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)   # k komşulu KNN modeli oluştur
            knn.fit(X_train, y_train)                   # modeli eğitim verisiyle eğit
            y_pread = knn.predict(X_test)               # test verisiyle tahmin yap
            accuracy = accuracy_score(y_test, y_pread)  # doğruluk skorunu hesapla
            accuracy_value.append(accuracy)             # doğruluk değerini listeye ekle
            k_value.append(k)                           # k değerini listeye ekle

        # Her k değeri için doğruluk skorlarını ekrana yazdırıyoruz
        print(accuracy_value)

        # K değerine göre doğruluk grafiğini çiziyoruz
        plt.figure()
        plt.plot(k_value, accuracy_value, marker="o", linestyle="-")  # k değerleri vs doğruluk
        plt.title("K değerlerine göre doğruluk")  # başlık
        plt.xlabel("K değerleri")                 # x ekseni etiketi
        plt.ylabel("doğruluk")                    # y ekseni etiketi
        plt.xticks(k_value)                       # x ekseninde her k değeri gösterilsin
        plt.grid(True)                            # arka plan ızgarası ekle
        plt.show()                                # grafiği göster

# Sınıfı çağırarak çalıştırıyoruz
Machine_learn1_5()
