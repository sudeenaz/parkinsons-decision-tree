import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Veri setini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

# Veri seti hakkında bilgi
print("\nVeri Seti Hakkında Detaylı Bilgi:", flush=True)
print("=" * 80, flush=True)
print(df.info(), flush=True)
print("=" * 80, flush=True)

# Eksik değer analizi
print("Eksik Değer Analizi:", flush=True)
print("=" * 80, flush=True)
print(df.isnull().sum(), flush=True)
print("=" * 80, flush=True)

# Veri seti özeti
print(f"Toplam Örnek Sayısı: {len(df)}", flush=True)
print(f"Toplam Özellik Sayısı: {len(df.columns)}", flush=True)
print("Özellikler:", flush=True)
print(df.columns.tolist(), flush=True)
print("Veri Tipleri:", flush=True)
print(df.dtypes, flush=True)

# Özellikler ve hedef değişkeni ayır
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Veriyi ölçeklendir 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Karar Ağacı modelini oluştur ve eğit
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Tahminler
y_pred = dt.predict(X_test)

# Model performans metrikleri
print("\nKarar Ağacı Sınıflandırıcı Sonuçları:", flush=True)
print(f"Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2f}", flush=True)
print("Sınıflandırma Raporu:", flush=True)
print(classification_report(y_test, y_pred, target_names=['Parkinson Yok', 'Parkinson Var']), flush=True)

# Karışıklık matrisi
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.savefig('karar_agaci_karisiklik_matrisi.png')
plt.close()

# ROC eğrisi
y_pred_proba = dt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Karar Ağacı ROC Eğrisi')
plt.legend(loc="lower right")
plt.savefig('karar_agaci_roc_egrisi.png')
plt.close()

# Özellik önemlilikleri
feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': dt.feature_importances_
})
feature_importance = feature_importance.sort_values('Önem', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Önem', y='Özellik', data=feature_importance)
plt.title('Özellik Önemlilikleri')
plt.tight_layout()
plt.savefig('ozellik_onemlilikleri.png')
plt.close()

# Özellik önemliliklerini yazdır
print("Tüm Özelliklerin Önem Dereceleri:")
print(feature_importance.to_string(index=False))

# En önemli 5 özelliği yazdır
print("En Önemli 5 Ses Özelliği:")
print(feature_importance.head().to_string(index=False))
