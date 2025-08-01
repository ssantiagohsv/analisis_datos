import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# --- 1️⃣ Preparación del entorno ---

# Cargar los datos desde el archivo CSV
# Es importante asegurarse de que el archivo esté en la ubicación correcta
try:
    df = pd.read_csv('data/supervised_dataset.csv', parse_dates=['Fecha'], decimal=',')
    df.set_index('Fecha', inplace=True)
except FileNotFoundError:
    print("Error: No se encontró el archivo 'supervised_dataset.csv'.")
    print("Por favor, asegúrate de que esté en la ruta 'data/'.")
    exit()

# Definir las variables (features) y la variable objetivo (target)
features = [
    'CIDIA_m3dia',
    'PMA_Kgcm2',
    'Zf',
    'Fpv',
    'SNDVEL_promedio',
    'Q_P_ratio',
    'Q_Z_ratio',
    'Q_Fpv_ratio'
]
target = 'Etiqueta'

X = df[features]
y = df[target]

# Dividir los datos en conjuntos de entrenamiento (70%) y prueba (30%)
# El parámetro `stratify=y` asegura que la proporción de clases sea la misma en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=== Preparación de Datos Completada ===")
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
print("---")

# --- 2️⃣ Entrenamiento del modelo ---

# Usar RandomForestClassifier como modelo base (baseline)
# Se ajustan hiperparámetros básicos para un mejor rendimiento
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Entrenar el modelo con el conjunto de entrenamiento
clf.fit(X_train, y_train)

# Opcional: Validación con cross-validation para evaluar la robustez del modelo
# Esto se realiza en el conjunto de entrenamiento y no afecta la evaluación final
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print("=== Validación Cruzada (Accuracy) ===")
print(f"Puntuaciones de CV: {scores}")
print(f"Accuracy promedio de CV: {np.mean(scores):.4f}")
print("---")

# --- 3️⃣ Evaluación del modelo ---

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular y mostrar las métricas de evaluación
print("=== Métricas de Evaluación ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("---")

# Extraer la importancia de las variables del modelo Random Forest
feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)

# Guardar un gráfico de la importancia de las variables
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh')
plt.title('Importancia de las Variables (Feature Importance)')
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.tight_layout()
plt.savefig('output/figures_supervised/feature_importances.png')
plt.close()

# --- 4️⃣ Interpretabilidad (opcional, pero recomendado) ---

# Usar SHAP para una interpretabilidad más profunda
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot resumen global de SHAP (importancia media)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Resumen de Importancia SHAP (Global)')
plt.savefig('output/figures_supervised/shap_summary_bar.png', bbox_inches='tight')
plt.close()

print("Gráfico de importancia de variables SHAP guardado en 'output/figures_supervised/shap_summary_bar.png'")
print("---")

# --- 5️⃣ Guardar resultados ---

# Guardar el modelo entrenado
joblib.dump(clf, 'output/rf_classifier.pkl')
print("Modelo guardado en: output/rf_classifier.pkl")

# Guardar las métricas en un archivo de texto
with open('output/classification_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
print("Métricas guardadas en: output/classification_metrics.txt")

print("\n✅ Script supervisado completado. ¡A analizar los resultados!")