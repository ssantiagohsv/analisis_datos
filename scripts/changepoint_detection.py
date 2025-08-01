import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os

DATA_PATH = 'data/raw_data.csv'
OUTPUT_PATH = 'output/figures_changepoint/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Leer datos y preparar
df = pd.read_csv(DATA_PATH, delimiter=',', encoding='utf-8-sig')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

# Limpieza formato numérico para las variables originales y SNDVEL5 a SNDVEL8
num_cols = ['CIDIA_m3dia', 'FRECUENCIA_Hz', 'PMA_Kgcm2', 'CIDIA93_m3dia',
            'SNDVEL5_ms', 'SNDVEL6_ms', 'SNDVEL7_ms', 'SNDVEL8_ms']

for col in num_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = df[col].astype(float)

# Crear promedio velocidad sonido siempre, antes de definir variables
df['SNDVEL_promedio'] = df[['SNDVEL5_ms', 'SNDVEL6_ms', 'SNDVEL7_ms', 'SNDVEL8_ms']].mean(axis=1)

# Variables a analizar (ya incluye SNDVEL_promedio)
variables = [
    'CIDIA_m3dia',
    'FRECUENCIA_Hz',
    'PMA_Kgcm2',
    'SNDVEL_promedio'
]

# Función para detectar y graficar rupturas
def detect_changepoints(series, model="rbf", pen=10):
    algo = rpt.Pelt(model=model).fit(series.values)
    result = algo.predict(pen=pen)
    return result

for var in variables:
    if var not in df.columns:
        continue
    series = df[var].dropna()
    change_points = detect_changepoints(series, pen=10)

    plt.figure(figsize=(15,5))
    plt.plot(series.index, series.values, label=var)
    for cp in change_points[:-1]:
        plt.axvline(series.index[cp], color='r', linestyle='--', label='Punto de cambio' if cp == change_points[0] else "")
    plt.title(f'Detección de rupturas - {var}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}{var}_changepoints.png")
    plt.close()

print("Detección de puntos de cambio finalizada. Revisá las imágenes en:", OUTPUT_PATH)
