import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Learn:
    def __init__(self):
        # Veri setini yükle
        url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
        data = pd.read_csv(url)
        print(data.head())
        print(data.info())
        print("---------------------------------------")

        pr1=data["Pregnancies"].mean()
        pr2=data["Glucose"].mean()
        pr3=data["BloodPressure"].mean()
        pr4=data["SkinThickness"].mean()
        pr5=data["Insulin"].mean()
        pr6=data["BMI"].mean()
        pr7=data["DiabetesPedigreeFunction"].mean()
        pr8=data["Age"].mean()
        

        avarenge=[pr1,pr2,pr3,pr4,pr5,pr6,pr7,pr8,]

        # Özellikler ve hedef
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]

        # Train/test ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=45, test_size=0.3
        )

        # KNN modeli
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

         
        features = ["Glucose", "BMI"]
        for i in range(10):  # ilk 10 hastayı göster
            plt.bar([f + f"_{i}" for f in features], X_train.iloc[i][features], color="skyblue")

        plt.ylabel("Değer")
        plt.title("Glucose ve BMI Değerleri (İlk 10 Hasta)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
        # Test setinde doğruluk
        y_pred_test = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Model doğruluk oranı: {accuracy:.2f}")

        # Örnek hasta
        sample = [[3, 120, 70, 25, 100, 32.0, 0.5, 28]]
        y_pred_sample = knn.predict(sample)

        if y_pred_sample[0] == 1:
            print("Örnek hasta için: Diyabet riski VAR")
        else:
            print("Örnek hasta için: Diyabet riski YOK")
        features = [
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DPF","Age"
        ]

        plt.bar(features, sample[0], color="red", alpha=0.7, label="Hasta")
        plt.plot(features, avarenge, color="blue", marker="o", label="Ortalama")  # karşılaştırma için
        plt.xticks(rotation=45)
        plt.ylabel("Değer")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()




# Çalıştır
Learn()
