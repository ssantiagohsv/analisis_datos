# scripts/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = 'data/raw_data.csv'
OUTPUT_PATH = 'output/figures/'

# Leer CSV
df = pd.read_csv(DATA_PATH, delimiter=',', encoding='utf-8-sig')

# Parsear Fecha
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

# Limpiar nombres columnas si queda algún espacio
df.columns = df.columns.str.strip()

# Columnas a limpiar y usar
variables = [
    'CIDIA_m3dia',
    'CIDIA93_m3dia',
    'FRECUENCIA_Hz',
    'PMA_Kgcm2',
    'PAB_Kgcm2',
    'Fpv',
    'Zb',
    'Zf',
    'SNDVEL5_ms',
    'SNDVEL6_ms',
    'SNDVEL7_ms',
    'SNDVEL8_ms'
]

# Limpiar separador miles y decimal en todas las columnas indicadas
for col in variables:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)  # quitar puntos miles
    df[col] = df[col].str.replace(',', '.', regex=False)             # cambiar coma decimal por punto
    df[col] = df[col].astype(float)


# Crear promedio velocidad del sonido
df['SNDVEL_promedio'] = df[['SNDVEL5_ms', 'SNDVEL6_ms', 'SNDVEL7_ms', 'SNDVEL8_ms']].mean(axis=1)

# Mostrar estadística descriptiva básica
print(df[variables + ['SNDVEL_promedio']].describe())

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Graficar series de tiempo
for var in variables + ['SNDVEL_promedio']:
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df[var])
    plt.title(f'Serie de tiempo - {var}')
    plt.xlabel('Fecha')
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{var}_serie.png'))
    plt.close()

# Heatmap de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df[variables + ['SNDVEL_promedio']].corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_heatmap.png'))
plt.close()