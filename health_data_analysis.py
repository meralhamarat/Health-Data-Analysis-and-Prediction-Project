import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Veri Yükleme
file_path = 'health_data.csv'
data = pd.read_csv(file_path)

# Veri İnceleme
print("Veri setinin ilk 5 satırı:")
print(data.head())

print("\nVeri setinin istatistiksel özetleri:")
print(data.describe())

# Eksik Değerlerin Kontrolü
print("\nEksik değerlerin kontrolü:")
print(data.isnull().sum())

# Veri Temizleme
# Eksik değerleri olan satırları düşürme (örneğin)
data = data.dropna()

# Veriyi Görselleştirme
plt.figure(figsize=(10, 6))
sns.pairplot(data, hue='Target')
plt.show()

# Özellik ve Hedef Değişkenlerin Ayrılması
X = data.drop('Target', axis=1)
y = data['Target']

# Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Makine Öğrenimi Modeli Eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modelin Değerlendirilmesi
y_pred = model.predict(X_test)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

# Özellik Önem Düzeylerinin Görselleştirilmesi
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Özellik Önem Düzeyleri")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()
