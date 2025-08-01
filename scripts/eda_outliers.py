# scripts/eda_outliers.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = 'data/raw_data.csv'
OUTPUT_PATH = 'output/figures_eda_outliers/'

# Leer y preparar datos
df = pd.read_csv(DATA_PATH, delimiter=',', encoding='utf-8-sig')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

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

# Limpieza formato numérico
for col in variables:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = df[col].astype(float)

os.makedirs(OUTPUT_PATH, exist_ok=True)

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index, outliers

outliers_summary = {}

for var in variables:
    plt.figure(figsize=(14,6))
    
    # Histograma
    plt.subplot(1,2,1)
    sns.histplot(df[var], kde=True, color='skyblue')
    plt.title(f'Histograma de {var}')
    
    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=df[var], color='lightcoral')
    plt.title(f'Boxplot de {var}')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}{var}_dist_boxplot.png")
    plt.close()
    
    # Detectar outliers
    idx_outliers, out_vals = detect_outliers_iqr(df[var])
    outliers_summary[var] = len(out_vals)
    
print("Resumen de outliers detectados (1.5*IQR):")
for var, count in outliers_summary.items():
    print(f"  {var}: {count} outliers")

