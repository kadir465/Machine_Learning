# 📊 Machine Learning – Algorithm Examples

Bu repository, **makine öğrenimi algoritmalarını temel seviyede öğrenmek ve uygulamak** amacıyla hazırlanmış örnek projeleri içermektedir. Çalışmalar özellikle **KNN (K-Nearest Neighbors)** ve **Decision Tree (Karar Ağaçları)** algoritmalarına odaklanır.

Her klasör, ilgili algoritmanın farklı senaryolarda uygulanmasını, görselleştirilmesini ve Python kodu ile mantığının anlaşılmasını hedefler.

---

## 📁 Proje Yapısı

```text
Machine_Learning/
└── machine_learninig/
    └── machine_learn/
        ├── dt/
        │   ├── machine_learning_DT1.py
        │   ├── learn_DT0.png
        │   └── learn_DT1.png
        │
        ├── knn/
        │   ├── machine_learning_KNN1.py
        │   ├── machine_learning_KNN2.py
        │   ├── machine_learning_KNN15.py
        │   └── learning1.png
        │
        └── knn_example/
            ├── knn_example1.py
            ├── knn_example2.ipynb
            ├── knn_and_tree_example.ipynb
            └── knn_and_tree_example2.ipynb
```

---

## 🎯 Projenin Amacı

Bu repository'nin temel amaçları şunlardır:

* Makine öğrenimi algoritmalarının **çalışma mantığını kavramak**
* KNN ve Decision Tree algoritmalarını Python ile **sıfırdan uygulamak**
* Model çıktılarının **görselleştirme ile yorumlanmasını sağlamak**
* Teorik bilgiyi **küçük ve anlaşılır kod örnekleriyle** pekiştirmek

Bu repo özellikle **öğrenme odaklı** hazırlanmıştır, üretim (production) amaçlı değildir.

---

## 🧠 Klasör Detayları

### 📂 `dt/` – Decision Tree (Karar Ağaçları)

Bu klasörde **Decision Tree algoritmasının** temel bir uygulaması yer alır.

* `machine_learning_DT1.py`

  * Karar ağacı modeli oluşturma
  * Eğitim ve tahmin işlemleri
  * Veriye göre karar mekanizmasının kurulması

* `learn_DT0.png`, `learn_DT1.png`

  * Karar ağacının veya öğrenme sürecinin görsel çıktıları
  * Modelin nasıl dallandığını anlamaya yardımcı olur

📌 **Amaç:** Karar ağaçlarının nasıl çalıştığını görsel ve kod üzerinden öğretmek.

---

### 📂 `knn/` – K-Nearest Neighbors (KNN)

Bu klasör, **KNN algoritmasının farklı K değerleri ile nasıl davrandığını** inceleyen Python scriptlerini içerir.

* `machine_learning_KNN1.py`
* `machine_learning_KNN2.py`
* `machine_learning_KNN15.py`

Bu dosyalarda:

* Farklı **K değerlerinin** model sonuçlarına etkisi

* Sınıflandırma mantığı

* En yakın komşu hesaplamaları

* `learning1.png`

  * KNN sonuçlarını veya veri dağılımını gösteren görsel çıktı

📌 **Amaç:** KNN algoritmasında K parametresinin model performansına etkisini göstermek.

---

### 📂 `knn_example/` – KNN ve Decision Tree Örnekleri

Bu klasör, hem **KNN** hem de **Decision Tree** algoritmalarının **örnekler ve notebooklar** üzerinden anlatıldığı daha detaylı çalışmaları içerir.

* `knn_example1.py`

  * Basit KNN uygulaması (script tabanlı)

* `knn_example2.ipynb`

  * Adım adım KNN anlatımı
  * Görselleştirmeler ve açıklamalar

* `knn_and_tree_example.ipynb`

* `knn_and_tree_example2.ipynb`

  * KNN ve Decision Tree algoritmalarının karşılaştırılması
  * Aynı veri seti üzerinde iki farklı yaklaşım

📌 **Amaç:** Algoritmalar arasındaki farkları uygulamalı olarak göstermek.

---

## 🛠 Kullanılan Teknolojiler

* **Python**
* **NumPy**
* **Scikit-Learn**
* **Matplotlib**
* **Jupyter Notebook**

---

## ▶️ Nasıl Çalıştırılır?

1. Repository'yi klonlayın:

```bash
git clone https://github.com/kadir465/Machine_Learning.git
```

2. Proje dizinine girin:

```bash
cd Machine_Learning/machine_learninig/machine_learn
```

3. Gerekli kütüphaneleri yükleyin:

```bash
pip install numpy scikit-learn matplotlib
```

4. Python dosyalarını çalıştırın:

```bash
python machine_learning_KNN1.py
```

veya notebook için:

```bash
jupyter notebook
```

---

## 🎓 Kimler İçin Uygun?

* Makine öğrenimine yeni başlayanlar
* Üniversite öğrencileri
* KNN ve Decision Tree algoritmalarını öğrenmek isteyenler
* Python ile ML pratiği yapmak isteyenler

---

## 👤 Geliştirici

**Kadir Emir Yücel**
Machine Learning • Python • Eğitim Odaklı Projeler

---

📌 *Bu repo, makine öğrenimi algoritmalarını sade, anlaşılır ve örneklerle öğretmeyi hedefler.*
