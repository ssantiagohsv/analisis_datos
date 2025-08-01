# supervised_classifier.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================
# CONFIGURACIÓN
# ============================

DATA_PATH = 'data/supervised_dataset.csv'
MODEL_PATH = 'output/rf_classifier.pkl'
METRICS_PATH = 'output/classification_metrics.txt'
FIGURES_PATH = 'output/figures_supervised/'
os.makedirs(FIGURES_PATH, exist_ok=True)

# ============================
# 1️⃣ CARGA DE DATOS
# ============================

df = pd.read_csv(DATA_PATH, parse_dates=['Fecha'], decimal=',')  # <-- Modificado aquí
df.set_index('Fecha', inplace=True)

# ============================
# 2️⃣ DEFINIR FEATURES Y TARGET
# ============================

features = [
    'CIDIA_m3dia',       # Q
    'PMA_Kgcm2',         # P
    'Zf',                # Z
    'Fpv',
    'SNDVEL_promedio',   # velocidad sonido
    'Q_P_ratio',
    'Q_Z_ratio',
    'Q_Fpv_ratio'
]

X = df[features]
y = df['Etiqueta']

# ============================
# 3️⃣ SPLIT TRAIN/TEST
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================
# 4️⃣ ENTRENAR CLASIFICADOR
# ============================

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

# ============================
# 5️⃣ EVALUACIÓN
# ============================

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("=== Accuracy ===")
print(acc)
print("\n=== Confusion Matrix ===")
print(cm)
print("\n=== Classification Report ===")
print(report)

# Guardar métricas
with open(METRICS_PATH, 'w') as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# ============================
# 6️⃣ IMPORTANCIA DE VARIABLES
# ============================

importances = clf.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title('Importancia de Variables - Random Forest')
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}feature_importances.png")
plt.close()

# ============================
# 7️⃣ GUARDAR MODELO
# ============================

joblib.dump(clf, MODEL_PATH)

print("\n✅ Entrenamiento y evaluación completados.")
print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Métricas guardadas en: {METRICS_PATH}")
print(f"Gráfico de importancia en: {FIGURES_PATH}feature_importances.png")
