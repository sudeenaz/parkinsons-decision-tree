🧠 Parkinson Hastalığı Sınıflandırması – Karar Ağacı Algoritması ile
Bu proje, Parkinson hastalığını sınıflandırmak amacıyla karar ağacı (Decision Tree) algoritması kullanılarak geliştirilmiştir. Veri madenciliği teknikleri ve Python programlama dili kullanılarak veri ön işleme, model eğitimi, değerlendirme, ve görselleştirme adımları gerçekleştirilmiştir.

🔍 Projenin Amacı
Parkinson hastalığına sahip bireyleri, ses kayıtları üzerinden elde edilen özelliklerle doğru bir şekilde sınıflandırmak ve bu sınıflandırma sürecinde karar ağacı algoritmasının performansını değerlendirmektir.

🗂 Kullanılan Veri Seti
Veri Kümesi: Parkinson’s Disease Data Set – UCI Machine Learning Repository

Özellikler: Toplamda 23 özellik bulunmakta olup bireylerin ses verilerinden türetilmiştir (örneğin MDVP:Fo(Hz), Jitter(%), Shimmer, vb.).

Hedef Değişken:

status: 1 → Parkinson hastası

status: 0 → Sağlıklı birey

⚙️ Kullanılan Teknolojiler ve Kütüphaneler
Python 3.x

pandas, numpy – Veri işleme

matplotlib, seaborn – Görselleştirme

scikit-learn – Makine öğrenimi ve model değerlendirme

🔄 Proje Aşamaları
1. Veri Yükleme ve İnceleme
Veri, UCI üzerinden doğrudan okunarak yüklendi.

df.info() ve df.describe() komutlarıyla veri analizi yapıldı.

Eksik değer kontrolü gerçekleştirildi.

2. Veri Ön İşleme
name sütunu kaldırıldı (anlamsız bilgi içeriyor).

status hedef değişken olarak ayrıldı.

StandardScaler ile sayısal özellikler standartlaştırıldı.

Veri eğitim (%80) ve test (%20) olarak bölündü.

3. Model Oluşturma ve Eğitme
DecisionTreeClassifier ile model oluşturuldu.

Eğitim verisi ile model eğitildi.

Test verileri ile tahmin yapıldı.

4. Model Değerlendirme
Doğruluk (Accuracy)

Sınıflandırma Raporu (Precision, Recall, F1-Score)

Karışıklık Matrisi (Confusion Matrix)

ROC Eğrisi ve AUC Skoru

5. Özellik Önem Dereceleri
feature_importances_ yardımıyla hangi özelliklerin karar verme sürecinde ne kadar etkili olduğu analiz edildi.

📊 Görselleştirme
Karışıklık Matrisi: Modelin doğru ve yanlış sınıflandırmalarını gösterir.

ROC Eğrisi: Modelin pozitif sınıfı ayırt etme başarımını görsel olarak sunar.

Özellik Önem Grafiği: Karar ağacının en çok dikkate aldığı öznitelikler gösterilir.

✅ Elde Edilen Sonuçlar
Model yüksek doğruluk oranı ile çalıştı (örneğin: %90+ doğruluk).

ROC eğrisi altında kalan alan (AUC) değeri 0.9 üzerindeydi.

En belirleyici özellikler arasında MDVP:Fo(Hz) ve spread1 gibi frekans ve titreme özellikleri yer aldı.

📁 Proje Dosya Yapısı

📦 parkinsons-decision-tree/
├── parkinsons_decision_tree.ipynb     # Proje Jupyter defteri
├── README.md                          # Bu dosya
├── data/
│   └── parkinsons.data                # Veri seti (isteğe bağlı)
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
🧪 Nasıl Çalıştırılır?
Gerekli kütüphaneleri yükleyin:

pip install pandas numpy matplotlib seaborn scikit-learn
Jupyter Notebook ile dosyayı açın:


jupyter notebook parkinsons_decision_tree.ipynb
Hücreleri sırasıyla çalıştırarak modeli eğitin ve çıktıları gözlemleyin.

💡 Gelecekte Neler Eklenebilir?
Karar ağacının derinliği ve dallanma kriterleri üzerinde hiperparametre optimizasyonu

Diğer sınıflandırma algoritmaları (Random Forest, SVM) ile karşılaştırma

Web tabanlı bir arayüz ile modeli sunma (örneğin Flask veya Streamlit ile)

📚 Kaynaklar
UCI Parkinson Dataset: https://archive.ics.uci.edu/ml/datasets/parkinsons

Scikit-learn Documentation: https://scikit-learn.org/

Python Data Science Handbook – Jake VanderPlas