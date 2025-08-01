import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os

# Paths
DATA_PATH = 'data/raw_data.csv'
OUTPUT_PATH = 'output/figures_changepoint/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Leer datos
df = pd.read_csv(DATA_PATH, delimiter=',', encoding='utf-8-sig')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

# Variables a analizar
variables = [
    'CIDIA_m3dia',
    'FRECUENCIA_Hz',
    'PMA_Kgcm2',
    'SNDVEL_promedio'
]

# Limpieza numérica
for col in variables + ['CIDIA93_m3dia', 'SNDVEL5_ms', 'SNDVEL6_ms', 'SNDVEL7_ms', 'SNDVEL8_ms']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = df[col].astype(float)

# Crear SNDVEL_promedio si no existe
if 'SNDVEL_promedio' not in df.columns:
    df['SNDVEL_promedio'] = df[['SNDVEL5_ms', 'SNDVEL6_ms', 'SNDVEL7_ms', 'SNDVEL8_ms']].mean(axis=1)

# Función detección
def detect_changepoints(series, model="rbf", pen=10):
    algo = rpt.Pelt(model=model).fit(series.values)
    result = algo.predict(pen=pen)
    return result

# Penalizaciones
pen_values = [5, 10, 20]

# Loop variables
for var in variables:
    if var not in df.columns:
        continue

    series = df[var].dropna()

    plt.figure(figsize=(15, 5))
    plt.plot(series.index, series.values, label=var)

    for i, pen in enumerate(pen_values):
        change_points = detect_changepoints(series, pen=pen)
        for cp in change_points[:-1]:
            plt.axvline(series.index[cp], linestyle='--', color=f'C{i+1}', alpha=0.7,
                        label=f'Pen={pen}' if cp == change_points[0] else "")

    plt.title(f'Detección de rupturas múltiples - {var}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}{var}_multi_changepoints.png")
    plt.close()

print("Análisis multi-penalización completado. Revisá las imágenes en:", OUTPUT_PATH)
