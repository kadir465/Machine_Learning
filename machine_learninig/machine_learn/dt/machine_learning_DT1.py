from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class learn2():
    def __init__(self):
        # Veri setini yükle
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Eğitim ve test verisine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=35
        )

        # Karar ağacı modeli
        tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)

        # Modeli eğit
        tree_clf.fit(X_train, y_train)

        # Tahmin yap
        y_pred = tree_clf.predict(X_test)

        # Doğruluk hesapla
        accuracy = accuracy_score(y_test, y_pred)
        print("Modelin doğruluk değeri:", accuracy)

        # Confusion matrix yazdır
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        plt.figure(figsize=(6,4))
        sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=iris.target_names,yticklabels=iris.target_names)
        plt.xlabel("Predicted")
        plt.ylabel( "Actual")
        plt.title("confusion matrix")
        plt.savefig("learn_DT0.png")
        plt.show()
        # Karar ağacını çiz
        plt.figure(figsize=(15, 10))
        plot_tree(
            tree_clf,
            filled=True,
            feature_names=iris.feature_names,
            class_names=list(iris.target_names),
        )
        plt.savefig("learn_DT1.png")
        plt.show()

        festure_importances=tree_clf.feature_importances_
        feature_names=iris.feature_names
        festure_importances_sorted=sorted(zip(festure_importances,feature_names),reverse=True)
        for importance, feature_names in festure_importances_sorted:
            print(f"{feature_names}:{importance}")


learn2()
