import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar modelo
clf = joblib.load('output/rf_classifier.pkl')

# Cargar datos
df = pd.read_csv('data/supervised_dataset.csv', parse_dates=['Fecha'], decimal=',')
df.set_index('Fecha', inplace=True)

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

X = df[features]
y = df['Etiqueta']

# Split igual que antes para obtener test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Crear explainer SHAP para Random Forest
explainer = shap.TreeExplainer(clf)

# Calcular valores SHAP
shap_values = explainer.shap_values(X_test)

# Plot resumen global (importancia media)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Plot resumen detallado
shap.summary_plot(shap_values, X_test)

# Opcional: guardar imagen resumen global
plt.savefig('output/figures_supervised/shap_summary_bar.png')
