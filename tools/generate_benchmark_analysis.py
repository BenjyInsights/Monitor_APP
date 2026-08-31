#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_benchmark_analysis.py

Suite integral de análisis para el benchmark de eficiencia energética en Deep Learning.

DESCRIPCIÓN GENERAL:
Este script consolida el análisis exhaustivo del banco de pruebas, generando:
  - 10 tablas CSV con métricas comparativas (Tablas 1-10)
  - 9 gráficas PDF vectoriales de alta resolución (Gráficas 1-9)
  - Metadatos y archivos de contexto reproducible

CARACTERÍSTICAS:
  ✓ Comparativa Hardware: Control vs Zeus (power capping NVIDIA)
  ✓ Comparativa Software: Zeus_Only vs Full_Optimized (con Early Stopping Scheduler)
  ✓ Ranking Global: Por eficiencia energética (J/muestra)
  ✓ Escalabilidad: Análisis de batch size sweep (32, 64, 128, 256)
  ✓ Diferenciación arquit. (ligeras/medias/pesadas)
  ✓ Normalización CO₂: gCO₂ / % exactitud alcanzada
  ✓ Estadísticas: Media, Std Dev, IC 95% para métricas clave
  ✓ Significancia: Tests Mann-Whitney U (no paramétricos)
  ✓ Impacto Geográfico: Emisiones CO₂ por 12 regiones (Ember 2025)
  ✓ Visualización Pareto: Frontera de eficiencia energética real

DATOS DE ENTRADA:
  - benchmark_master_dataset.json (agregación de 420 runs: 84 configuraciones × 5 repeticiones)
  - layer_profile CSV (desagregación por capas en ResNet18)

SALIDA:
  - results/csv/         (10 tablas CSV de análisis)
  - results/figuras/     (9 gráficas PDF; redirigible con MONIAENERGY_FIGURES_DIR)
  - results/metadata/    (metadatos reproducibles)

CORRECCIONES v2 (2026-04-14):
  ✓ Frontera de Pareto real (no regresión simple)
  ✓ Valores negativos en gráfica_6 (casos ganar-ganar)
  ✓ Heatmap CO₂ geografico en gráfica_9
  ✓ Tests de significancia Mann-Whitney U robustos
  ✓ Paleta UNED institucional homogénea en todas las gráficas
  ✓ Ruta adaptativa para layer_profile (1er CSV disponible)

DEPENDENCIAS:
  - pandas, numpy, matplotlib, seaborn, scipy

USO:
  $ python generate_benchmark_analysis.py

REFACTORIZACIÓN (2026-04-15):
  ✓ Consolidación de tres archivos (generate_tfm_stats.py, extended, sprint2)
  ✓ Extracción de configuración a analysis_utils.py
  ✓ Docstrings profesionales (Google Style)
  ✓ Validación de entrada/salida
  ✓ Manejo robusto de errores y casos edge

Autor: Senior Software Architect
Fecha: 2026-04-15
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from datetime import datetime

# Importar configuración centralizada desde analysis_utils
from analysis_utils import (
    BASE_DIR, INPUT_FILE, OUTPUT_CSV, OUTPUT_PLOTS, OUTPUT_METADATA,
    UNED_COLORS, MODEL_COLORS, CO2_FACTORS,
    create_output_directories, validate_output_paths, load_benchmark_data,
    calculate_ci, pareto_frontier
)

# Configuración estética global para matplotlib
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 19
plt.rcParams['axes.labelsize'] = 19
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17


# ============================================================================
# GENERACIÓN DE TABLAS CSV (10 TABLAS DE ANÁLISIS)
# ============================================================================

def _paired_savings_std(baseline_df, treated_df) -> float:
    """Sigma inter-repetición del ahorro energético.

    Calcula el ahorro (%) repetición a repetición emparejando por índice de
    repetición y devuelve su desviación estándar muestral. Es la incertidumbre
    que corresponde reportar junto a un ahorro promediado sobre repeticiones;
    no debe confundirse con la dispersión entre épocas de una única corrida.
    """
    paired = []
    for rep in sorted(set(baseline_df['Rep']) & set(treated_df['Rep'])):
        b = baseline_df[baseline_df['Rep'] == rep]['EnergyJ'].mean()
        t = treated_df[treated_df['Rep'] == rep]['EnergyJ'].mean()
        if b > 0:
            paired.append(((b - t) / b) * 100)
    return float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0


def generate_tabla_zeus_impact(df):
    """
    Tabla 1: Impacto de Hardware Optimization (Zeus power capping).
    
    Compara Control vs Zeus_Only (sin Early Stopping) para aislar
    el efecto del control de potencia NVIDIA Zeus del efecto del scheduler.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla con columnas de energía, accuracy y ahorros.
    
    CSV Output:
        tabla_1_impacto_zeus.csv
    """
    mask = (df['EarlyStopping'] == False)
    zeus_df = df[mask].copy()

    result = []
    for model in sorted(zeus_df['Model'].unique()):
        for batch in sorted(zeus_df['Batch'].unique()):
            subset = zeus_df[(zeus_df['Model'] == model) & (zeus_df['Batch'] == batch)]
            control = subset[subset['Mode'] == 'Control']
            zeus    = subset[subset['Mode'] == 'Zeus_Only']

            if not control.empty and not zeus.empty:
                # Media sobre las repeticiones disponibles (5 por configuración).
                ctrl_energy = control['EnergyJ'].mean()
                zeus_energy = zeus['EnergyJ'].mean()
                ctrl_acc    = control['Accuracy'].mean()
                zeus_acc    = zeus['Accuracy'].mean()
                savings     = ((ctrl_energy - zeus_energy) / ctrl_energy) * 100
                # Diferencia de exactitud en puntos porcentuales (no relativa).
                acc_change_pp = (zeus_acc - ctrl_acc) * 100

                savings_std = _paired_savings_std(control, zeus)

                result.append({
                    'Modelo':                  model,
                    'BatchSize':               batch,
                    'N_Reps':                  int(min(len(control), len(zeus))),
                    'Control_Energia_J':       round(ctrl_energy, 2),
                    'Control_Energia_Std':     round(control['EnergyJ'].std(ddof=1), 2),
                    'Zeus_Energia_J':          round(zeus_energy, 2),
                    'Zeus_Energia_Std':        round(zeus['EnergyJ'].std(ddof=1), 2),
                    'Ahorro_Energetico_%':     round(savings, 2),
                    'Ahorro_Std_%':            round(savings_std, 2),
                    'Control_Accuracy':        round(ctrl_acc, 4),
                    'Zeus_Accuracy':           round(zeus_acc, 4),
                    'CambioAccuracy_pp':       round(acc_change_pp, 2),
                })

    result_df = pd.DataFrame(result)
    result_df.to_csv(OUTPUT_CSV / "tabla_1_impacto_zeus.csv", index=False)
    print("✅ Tabla 1: Impacto Zeus (Hardware Optimization)")
    return result_df


def generate_tabla_ees_impact(df):
    """
    Tabla 2: Impacto de Software Optimization (Early Stopping Scheduler).
    
    Compara Zeus_Only vs Full_Optimized (Zeus + EES) para aislar
    el efecto del scheduler de terminación anticipada.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla con energía, épocas completadas y pérdida de exactitud.
    
    CSV Output:
        tabla_2_impacto_ees.csv
    """
    result = []
    for model in sorted(df['Model'].unique()):
        for batch in sorted(df['Batch'].unique()):
            subset   = df[(df['Model'] == model) & (df['Batch'] == batch)]
            zeus_only = subset[subset['Mode'] == 'Zeus_Only']
            full_opt  = subset[subset['Mode'] == 'Full_Optimized']

            control   = subset[subset['Mode'] == 'Control']

            if not zeus_only.empty and not full_opt.empty:
                # Media sobre las repeticiones disponibles (5 por configuración).
                z_energy  = zeus_only['EnergyJ'].mean()
                f_energy  = full_opt['EnergyJ'].mean()
                z_epochs  = zeus_only['EpochsCompleted'].mean()
                f_epochs  = full_opt['EpochsCompleted'].mean()
                z_acc     = zeus_only['Accuracy'].mean()
                f_acc     = full_opt['Accuracy'].mean()
                ees_savings     = ((z_energy - f_energy) / z_energy) * 100
                epoch_reduction = ((z_epochs - f_epochs) / z_epochs) * 100 if z_epochs > 0 else 0
                # Pérdida de exactitud en PUNTOS PORCENTUALES (diferencia absoluta de
                # exactitudes), no en variación relativa: es la magnitud que se reporta
                # en el texto y la que corresponde al test de significancia.
                acc_loss_pp_zeus = (z_acc - f_acc) * 100
                acc_loss_pp_ctrl = ((control['Accuracy'].mean() - f_acc) * 100
                                    if not control.empty else float('nan'))

                result.append({
                    'Modelo':                 model,
                    'BatchSize':              batch,
                    'N_Reps':                 int(min(len(zeus_only), len(full_opt))),
                    'Zeus_Energia_J':         round(z_energy, 2),
                    'FullOpt_Energia_J':      round(f_energy, 2),
                    'Ahorro_EES_%':           round(ees_savings, 2),
                    'Ahorro_EES_Std_%':       round(_paired_savings_std(zeus_only, full_opt), 2),
                    'Zeus_Epochs':            round(z_epochs, 1),
                    'FullOpt_Epochs':         round(f_epochs, 1),
                    'FullOpt_Epochs_Std':     round(full_opt['EpochsCompleted'].std(ddof=1), 1),
                    'ReduccionEpocas_%':      round(epoch_reduction, 2),
                    'PerdidaAcc_pp_vsZeus':   round(acc_loss_pp_zeus, 2),
                    'PerdidaAcc_pp_vsControl': round(acc_loss_pp_ctrl, 2),
                })

    result_df = pd.DataFrame(result)
    result_df.to_csv(OUTPUT_CSV / "tabla_2_impacto_ees.csv", index=False)
    print("✅ Tabla 2: Impacto EES (Software Optimization)")
    return result_df


def aggregate_ordinal_grades(grades: pd.Series) -> str:
    """
    Agrega calificaciones ordinales (A++ a F) promediando su valor numérico,
    resolviendo de forma robusta los empates que pd.Series.mode() resuelve alfabéticamente.
    """
    if grades.empty:
        return 'N/A'
    
    mapping = {'F': 0, 'E': 1, 'D': 2, 'C': 3, 'B': 4, 'A': 5, 'A+': 6, 'A++': 7}
    rev_mapping = {v: k for k, v in mapping.items()}
    
    ints = [mapping.get(g) for g in grades if g in mapping]
    if not ints:
        return 'N/A'
        
    avg = np.mean(ints)
    return rev_mapping.get(int(np.round(avg)), 'N/A')



def generate_tabla_ranking_global(df):
    """
    Tabla 3: Ranking global de modelos por eficiencia energética.
    
    Ordena las arquitecturas por J/muestra (métrica de eficiencia normalizada).
    Incluye exactitud máxima, CO₂ total, EDP y energía grade.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Ranking ordenado de menor a mayor consumo energético.
    
    CSV Output:
        tabla_3_ranking_global.csv
    """
    # El ranking caracteriza la eficiencia en modo Control (línea de base),
    # coherente con la Figura de intensidad energética y con el barrido de lote.
    ranking = []
    for model in sorted(df['Model'].unique()):
        model_data = df[(df['Model'] == model) & (df['Mode'] == 'Control')]
        ranking.append({
            'Modelo':            model,
            'J_Sample_Promedio': round(model_data['JSample'].mean(), 4),
            'J_Sample_Std':      round(model_data['JSample'].std(), 4),
            'CO2_Total_g':       round(model_data['CO2g'].sum(), 2),
            'Accuracy_Max_%':    round(model_data['Accuracy'].max() * 100, 2),
            'Accuracy_Promedio_%': round(model_data['Accuracy'].mean() * 100, 2),
            'EDP_Promedio':      round(model_data['AvgEDP'].mean(), 2),
            'Grade_Modal':       aggregate_ordinal_grades(model_data['Grade']),
            'Total_Runs':        len(model_data),
        })

    ranking_df = pd.DataFrame(ranking).sort_values('J_Sample_Promedio')
    ranking_df.to_csv(OUTPUT_CSV / "tabla_3_ranking_global.csv", index=False)
    print("✅ Tabla 3: Ranking Global de Eficiencia")
    return ranking_df


def generate_tabla_batch_sweep(df):
    """
    Tabla 4: Análisis de escalabilidad con batch size sweep.
    
    Muestra cómo J/muestra varía con tamaños de batch (32, 64, 128, 256).
    Incluye throughput (muestras/Joule).
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla con energía y throughput por batch size.
    
    CSV Output:
        tabla_4_batch_sweep.csv
    """
    result = []
    for model in sorted(df['Model'].unique()):
        for batch in sorted(df['Batch'].unique()):
            subset  = df[(df['Model'] == model) & (df['Batch'] == batch)]
            control = subset[subset['Mode'] == 'Control']
            if not control.empty:
                # Media sobre las repeticiones; la sigma es INTER-repetición
                # (dispersión entre corridas independientes de la misma configuración).
                jsample = float(control['JSample'].mean())
                jstd    = float(control['JSample'].std(ddof=1)) if len(control) > 1 else 0.0
                energy  = float(control['EnergyJ'].mean())
                acc     = float(control['Accuracy'].mean())
                result.append({
                    'Modelo':                   model,
                    'BatchSize':                batch,
                    'N_Reps':                   int(len(control)),
                    'J_Sample':                 round(jsample, 4),
                    'J_Sample_Std':             round(jstd, 4),
                    'Energia_Total_J':          round(energy, 2),
                    'Accuracy_%':               round(acc * 100, 2),
                    'Throughput_Samples_Per_J': round(1 / jsample, 2) if jsample > 0 else 0,
                })

    batch_df = pd.DataFrame(result)
    batch_df.to_csv(OUTPUT_CSV / "tabla_4_batch_sweep.csv", index=False)
    print("✅ Tabla 4: Análisis de Batch Size Sweep")
    return batch_df


def generate_tabla_by_architecture(df):
    """
    Tabla 5: Análisis diferenciado por tipo de arquitectura.
    
    Agrupa los modelos en categorías (ligeras, medias, pesadas) y compara
    su eficiencia, exactitud y emisiones.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla comparativa por tipo arquitectónico.
    
    CSV Output:
        tabla_5_arquitecturas.csv
    """
    ligeras = ['MobileNetV2', 'EfficientNetB0']
    pesadas = ['VGG19', 'ResNet50', 'DenseNet121']

    arch_map = {}
    for m in df['Model'].unique():
        if m in ligeras:
            arch_map[m] = 'Ligera'
        elif m in pesadas:
            arch_map[m] = 'Pesada'
        else:
            arch_map[m] = 'Media'

    df = df.copy()
    df['Arquitectura'] = df['Model'].map(arch_map)

    result = []
    for arch in ['Ligera', 'Media', 'Pesada']:
        subset  = df[df['Arquitectura'] == arch]
        control = subset[subset['Mode'] == 'Control']
        if not control.empty:
            result.append({
                'Tipo_Arquitectura': arch,
                'J_Sample_Mean':    round(control['JSample'].mean(), 4),
                'J_Sample_Std':     round(control['JSample'].std(), 4),
                'Accuracy_Mean_%':  round(control['Accuracy'].mean() * 100, 2),
                'CO2_Total_g':      round(control['CO2g'].sum(), 2),
                'EDP_Mean':         round(control['AvgEDP'].mean(), 2),
                'Modelos':          len(control['Model'].unique()),
                'Total_Runs':       len(control),
            })

    arch_df = pd.DataFrame(result)
    arch_df.to_csv(OUTPUT_CSV / "tabla_5_arquitecturas.csv", index=False)
    print("✅ Tabla 5: Análisis por Arquitectura")
    return arch_df


def generate_tabla_co2_normalized(df):
    """
    Tabla 6: Normalización de CO₂ por porcentaje de exactitud alcanzada.
    
    Calcula gCO₂ / % exactitud para evaluar la eficiencia
    en términos de precisión lograda por unidad de contaminación.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla con CO₂ normalizado.
    
    CSV Output:
        tabla_6_co2_normalized.csv
    """
    result = []
    for model in sorted(df['Model'].unique()):
        for batch in sorted(df['Batch'].unique()):
            subset = df[(df['Model'] == model) & (df['Batch'] == batch)]
            for mode in subset['Mode'].unique():
                mode_data = subset[subset['Mode'] == mode]
                if not mode_data.empty:
                    co2        = mode_data['CO2g'].values[0]
                    acc_percent = mode_data['Accuracy'].values[0] * 100
                    co2_per_acc = co2 / acc_percent if acc_percent > 0 else np.nan
                    result.append({
                        'Modelo':                  model,
                        'BatchSize':               batch,
                        'Modo':                    mode,
                        'CO2_Total_g':             round(co2, 2),
                        'Accuracy_%':              round(acc_percent, 2),
                        'gCO2_per_AccuracyPercent': (round(co2_per_acc, 4)
                                                     if not np.isnan(co2_per_acc) else 'N/A'),
                    })

    co2_norm = pd.DataFrame(result)
    co2_norm.to_csv(OUTPUT_CSV / "tabla_6_co2_normalized.csv", index=False)
    print("✅ Tabla 6: CO₂ Normalizado por Accuracy")
    return co2_norm


def generate_tabla_estadisticas(df):
    """
    Tabla 7: Estadísticas descriptivas de métricas clave.
    
    Calcula media, desviación estándar, intervalo de confianza 95%,
    mínimo, máximo y tamaño de muestra para los KPIs principales.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla de estadísticas descriptivas.
    
    CSV Output:
        tabla_7_estadisticas.csv
    """
    result = []
    metrics = {
        'JSample':   'J/Muestra (J/sample)',
        'EnergyJ':   'Energía Total (J)',
        'Accuracy':  'Exactitud (Accuracy)',
        'CO2g':      'Emisiones CO₂ (g)',
        'AvgEDP':    'EDP Promedio (J·s)',
    }
    for metric, label in metrics.items():
        data = df[metric].dropna()
        if len(data) > 0:
            mean = data.mean()
            std  = data.std()
            ci_low, ci_high = calculate_ci(data)
            result.append({
                'Métrica':       label,
                'Media':         round(mean, 4),
                'Desv_Estandar': round(std, 4),
                'IC95_Bajo':     round(ci_low, 4),
                'IC95_Alto':     round(ci_high, 4),
                'Min':           round(data.min(), 4),
                'Max':           round(data.max(), 4),
                'N_Muestras':    len(data),
            })

    stats_df = pd.DataFrame(result)
    stats_df.to_csv(OUTPUT_CSV / "tabla_7_estadisticas.csv", index=False)
    print("✅ Tabla 7: Estadísticas Descriptivas")
    return stats_df


def generate_tabla_metadata():
    """
    Tabla 8: Metadatos del experiment reproducible.
    
    Documenta el contexto completo: dataset, modelos, hardware, herramientas,
    y propósito de la experimentación para garantizar reproducibilidad.
    
    Returns:
        pd.DataFrame: Tabla de metadatos.
    
    CSV Output:
        tabla_8_metadata.csv
    
    JSON Output:
        results/metadata/environment_info.json
    """
    metadata = {
        'Variable': [
            'Dataset',
            'Modelos Evaluados',
            'Total de Runs',
            'Batch Sizes',
            'Modos de Optimización',
            'Épocas por Run',
            'Fecha de Ejecución',
            'PyTorch + CUDA',
            'Métricas de Energía',
            'Métricas de Carbono',
            'GPU Monitorizado',
            'CPU Monitorizado',
            'Propósito',
        ],
        'Valor': [
            'CIFAR-10 (50.000 train / 10.000 test, 32×32 px, 3 canales)',
            'VGG19, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, ResNet18, ViT',
            '420 (7 modelos × 4 batch sizes × 3 modos × 5 repeticiones)',
            '[32, 64, 128, 256]',
            'Control, Zeus_Only (power capping), Full_Optimized (Zeus + EES)',
            '50 épocas fijas (Full_Optimized puede terminar antes por EES)',
            '2026-04-08 / 2026-04-13',
            'PyTorch 2.6 + CUDA 12.6',
            'NVIDIA NVML (GPU), Intel RAPL (CPU), psutil (memoria RAM)',
            'Factores de emisión IEA 2023 para 16 ubicaciones geográficas',
            'NVIDIA RTX 6000 Ada (48 GB VRAM) — vía pynvml',
            'Intel Xeon (compatible RAPL) — vía /sys/class/powercap/',
            'Framework moniaenergy — medición y optimización energética',
        ]
    }

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(OUTPUT_CSV / "tabla_8_metadata.csv", index=False)
    with open(OUTPUT_METADATA / "environment_info.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("✅ Tabla 8: Metadatos y Variables de Entorno")
    return metadata_df


def generate_tabla_significancia(df):
    """
    Tabla 9: Tests de significancia estadística (Mann-Whitney U).
    
    Valida formalmente que las diferencias observadas entre modos
    NO son artefactos de ruido, usando un test no paramétrico
    (robusto a distribuciones no normales).
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla con resultados de pruebas estadísticas.
    
    CSV Output:
        tabla_9_significancia.csv
    """
    result = []

    ctrl  = df[df['Mode'] == 'Control']['EnergyJ'].dropna()
    zeus  = df[df['Mode'] == 'Zeus_Only']['EnergyJ'].dropna()
    full  = df[df['Mode'] == 'Full_Optimized']['EnergyJ'].dropna()
    ctrl_acc = df[df['Mode'] == 'Control']['Accuracy'].dropna()
    full_acc = df[df['Mode'] == 'Full_Optimized']['Accuracy'].dropna()

    comparativas = [
        ('Control vs. Zeus_Only (Energía)',
         ctrl, zeus,
         'Control consume más energía que Zeus_Only',
         'greater'),
        ('Zeus_Only vs. Full_Optimized (Energía)',
         zeus, full,
         'Zeus_Only consume más energía que Full_Optimized',
         'greater'),
        ('Control vs. Full_Optimized (Energía)',
         ctrl, full,
         'Control consume más energía que Full_Optimized',
         'greater'),
        ('Control vs. Full_Optimized (Accuracy — igualdad)',
         ctrl_acc, full_acc,
         'Control y Full_Optimized tienen accuracy equivalente',
         'two-sided'),
    ]

    for label, x, y, hipotesis, alternative in comparativas:
        if len(x) > 0 and len(y) > 0:
            stat, p = mannwhitneyu(x, y, alternative=alternative)
            result.append({
                'Comparativa':         label,
                'Hipótesis alternativa': hipotesis,
                'Estadístico U':       round(stat, 2),
                'p-valor':             round(p, 6),
                'Significativo (α=0.05)': 'Sí' if p < 0.05 else 'No',
                'n_x':                 len(x),
                'n_y':                 len(y),
            })

    sig_df = pd.DataFrame(result)
    sig_df.to_csv(OUTPUT_CSV / "tabla_9_significancia.csv", index=False)
    print("✅ Tabla 9: Tests de Significancia Estadística (Mann-Whitney U)")
    return sig_df


def generate_tabla_co2_regional(df):
    """
    Tabla 10: Impacto geográfico (CO₂ estimado por región).
    
    Calcula emisiones de CO₂ para cada modelo × región usando factores
    de intensidad de carbono de la red eléctrica (Ember 2025).
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    
    Returns:
        pd.DataFrame: Tabla de CO₂ estimado por región.
    
    CSV Output:
        tabla_10_co2_regional.csv
    """
    control_bs256 = df[(df['Mode'] == 'Control') & (df['Batch'] == 256)].copy()

    # Una fila por modelo: energía media sobre las repeticiones. Antes se emitía
    # una fila por repetición, de modo que el CSV no coincidía con la tabla de la
    # memoria ni con el mapa de calor (que sí promediaban).
    mean_energy = control_bs256.groupby('Model')['EnergyJ'].mean()

    result = []
    for model, energy_j in mean_energy.items():
        energy_kwh = energy_j / 3_600_000  # J → kWh
        fila = {'Modelo': model, 'Energia_Media_J': round(float(energy_j), 2)}
        for region, factor in CO2_FACTORS.items():
            fila[region] = round(energy_kwh * factor * 1000, 2)  # kg → g
        result.append(fila)

    co2_regional = pd.DataFrame(result).sort_values('Modelo')
    co2_regional.to_csv(OUTPUT_CSV / "tabla_10_co2_regional.csv", index=False)
    print("✅ Tabla 10: CO₂ Regional por Arquitectura (batch 256, modo Control)")
    return co2_regional


# ============================================================================
# GENERACIÓN DE GRÁFICAS PDF VECTORIALES (9 GRÁFICAS)
# ============================================================================

def plot_pareto(df):
    """
    Gráfica 1: Frontera de Pareto real (Energía vs. Exactitud).
    
    Visualiza los puntos Pareto-óptimos (eficiencia máxima) conectados
    por una línea discontinua. Los puntos no están dominados por otros
    (no existe modelo que sea mejor en ambos objetivos simultáneamente).
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    # Un punto por configuración (modelo × lote), promediando las repeticiones:
    # 28 puntos en lugar de las 140 ejecuciones individuales.
    control_data = (df[df['Mode'] == 'Control']
                    .groupby(['Model', 'Batch'], as_index=False)[['EnergyJ', 'Accuracy']]
                    .mean())

    fig, ax = plt.subplots(figsize=(14, 8))

    # Scatter por modelo
    for model in sorted(control_data['Model'].unique()):
        model_data = control_data[control_data['Model'] == model]
        ax.scatter(model_data['EnergyJ'], model_data['Accuracy'] * 100,
                   label=model, s=120, alpha=0.85,
                   color=MODEL_COLORS.get(model, 'gray'),
                   edgecolors='black', linewidths=0.6, zorder=3)

    # Calcular y trazar frontera de Pareto real
    pareto_df = pareto_frontier(
        control_data[['EnergyJ', 'Accuracy', 'Model']].dropna(),
        x_col='EnergyJ', y_col='Accuracy'
    )
    if len(pareto_df) >= 2:
        ax.plot(pareto_df['EnergyJ'], pareto_df['Accuracy'] * 100,
                color=UNED_COLORS['RASPBERRY'], linewidth=2.0,
                linestyle='--', marker='D', markersize=8,
                label='Frontera de Pareto', zorder=4, alpha=0.85)

    ax.set_xlabel('Energía Total (Joules)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Exactitud Final (%)', fontsize=18, fontweight='bold')
    ax.set_title('Frontera de Pareto: Consumo Energético vs. Exactitud (Modo Control)',
                 fontsize=19, fontweight='bold')
    ax.legend(title='Modelo', fontsize=16, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_1_pareto.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_1_pareto.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 1: Frontera de Pareto")


def plot_energy_intensity(df):
    """
    Gráfica 2: Intensidad energética por modelo (J/muestra).
    
    Barras horizontales ordenadas de menor a mayor consumo.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    control_data = df[df['Mode'] == 'Control'].copy()
    intensity = control_data.groupby('Model')['JSample'].mean().sort_values()
    # Barras de error: dispersión entre las 20 ejecuciones en modo Control de cada
    # modelo (4 tamaños de lote × 5 repeticiones). Domina el efecto del lote.
    dispersion = control_data.groupby('Model')['JSample'].std(ddof=1).reindex(intensity.index)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors_sorted = [MODEL_COLORS.get(m, UNED_COLORS['UNED_MEDIUM']) for m in intensity.index]
    bars = ax.barh(intensity.index, intensity.values,
                   xerr=dispersion.values, capsize=6,
                   error_kw={'ecolor': 'black', 'elinewidth': 1.2, 'capthick': 1.2},
                   color=colors_sorted, edgecolor='black', linewidth=0.7)

    for bar, err in zip(bars, dispersion.values):
        width = bar.get_width()
        ax.text(width + err + intensity.max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.4f} J/muestra',
                ha='left', va='center', fontsize=16, fontweight='bold')

    ax.set_xlabel('Energía por Muestra (J/muestra)', fontsize=18, fontweight='bold')
    ax.set_title('Intensidad Energética por Modelo (Modo Control)',
                 fontsize=19, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, (intensity + dispersion.fillna(0)).max() * 1.30)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_2_intensidad_energetica.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_2_intensidad_energetica.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 2: Intensidad Energética")


def plot_batch_scaling(df):
    """
    Gráfica 3: Escalabilidad vs. batch size.
    
    Muestra cómo J/muestra varía con tamaños de batch para cada modelo.
    Incluye porcentaje de mejora de batch 32 a batch 256.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    control_data = df[df['Mode'] == 'Control'].copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    for model in sorted(control_data['Model'].unique()):
        # Una curva por modelo: media sobre las repeticiones de cada lote. Sin este
        # promediado se trazaban las 5 repeticiones como puntos superpuestos y la
        # línea saltaba en vertical sobre cada abscisa.
        model_data = (control_data[control_data['Model'] == model]
                      .groupby('Batch')['JSample'].mean().sort_index())
        if len(model_data) > 1:
            # Mejora % entre batch mínimo y máximo
            jsample_min = model_data.iloc[0]
            jsample_max = model_data.iloc[-1]
            pct_improvement = ((jsample_min - jsample_max) / jsample_min) * 100
            label = f"{model} (−{pct_improvement:.0f}% batch 32→256)"
        else:
            label = model
        ax.plot(model_data.index, model_data.values,
                marker='o', label=label, linewidth=2, markersize=8,
                color=MODEL_COLORS.get(model, 'gray'))

    ax.set_xlabel('Tamaño de Batch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Energía por Muestra (J/muestra)', fontsize=18, fontweight='bold')
    ax.set_title('Escalabilidad: Intensidad Energética vs. Tamaño de Batch',
                 fontsize=19, fontweight='bold')
    ax.legend(title='Modelo', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([32, 64, 128, 256])
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_3_batch_scaling.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_3_batch_scaling.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 3: Batch Size Scaling")


def plot_optimization_impact(df):
    """
    Gráfica 4: Impacto de optimizaciones (ahorro % respecto a Control).
    
    Compara Zeus_Only vs Full_Optimized. Barras negativas indican
    consumo superior al Control (casos raros).
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    # El ahorro se calcula POR CONFIGURACIÓN (modelo × lote) y luego se promedia sin
    # ponderar. Promediar directamente la energía de los 20 runs de un modelo daba una
    # media ponderada por energía, dominada por el lote 32, que contradecía los valores
    # de la tabla de impacto Zeus para el mismo experimento.
    result = []
    for model in sorted(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        zeus_per_cfg, full_per_cfg = [], []

        for batch in sorted(model_df['Batch'].unique()):
            cfg = model_df[model_df['Batch'] == batch]
            control = cfg[cfg['Mode'] == 'Control']
            if control.empty:
                continue
            ctrl_energy = control['EnergyJ'].mean()
            if ctrl_energy <= 0:
                continue
            zeus = cfg[cfg['Mode'] == 'Zeus_Only']
            full = cfg[cfg['Mode'] == 'Full_Optimized']
            if not zeus.empty:
                zeus_per_cfg.append(((ctrl_energy - zeus['EnergyJ'].mean()) / ctrl_energy) * 100)
            if not full.empty:
                full_per_cfg.append(((ctrl_energy - full['EnergyJ'].mean()) / ctrl_energy) * 100)

        if zeus_per_cfg or full_per_cfg:
            result.append({
                'Modelo': model,
                'Zeus': float(np.mean(zeus_per_cfg)) if zeus_per_cfg else 0.0,
                'Full': float(np.mean(full_per_cfg)) if full_per_cfg else 0.0,
            })

    result_df = pd.DataFrame(result).set_index('Modelo')

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(result_df))
    width = 0.3

    ax.bar(x - width / 2, result_df['Zeus'], width,
           label='Zeus Only (power capping)',
           color=UNED_COLORS['BLUE'], edgecolor='black', linewidth=0.6)
    ax.bar(x + width / 2, result_df['Full'], width,
           label='Full Optimized (Zeus + EES)',
           color=UNED_COLORS['UNED'], edgecolor='black', linewidth=0.6)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax.set_ylabel('Ahorro Energético respecto al Control (%)',
                  fontsize=18, fontweight='bold')
    ax.set_title('Impacto de las Optimizaciones Green AI (referencia: Modo Control)',
                 fontsize=19, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(result_df.index, rotation=30, ha='right')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Anotaciones de valor sobre cada barra
    for i, model in enumerate(result_df.index):
        z_val = result_df.loc[model, 'Zeus']
        f_val = result_df.loc[model, 'Full']
        ax.text(i - width / 2, z_val + (1 if z_val >= 0 else -3),
                f'{z_val:.1f}%', ha='center', va='bottom', fontsize=15, fontweight='bold')
        ax.text(i + width / 2, f_val + (1 if f_val >= 0 else -3),
                f'{f_val:.1f}%', ha='center', va='bottom', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_4_optimization_impact.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_4_optimization_impact.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 4: Impacto de Optimizaciones")


def plot_co2_footprint(df):
    """
    Gráfica 5: Huella de carbono acumulada (g CO₂eq) por modelo y modo.
    
    Compara Control, Zeus_Only y Full_Optimized usando paleta UNED.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    co2_data = df.groupby(['Model', 'Mode'])['CO2g'].sum().reset_index()

    modes   = ['Control', 'Zeus_Only', 'Full_Optimized']
    models  = sorted(co2_data['Model'].unique())
    x       = np.arange(len(models))
    width   = 0.25
    mode_colors = {
        'Control':        UNED_COLORS['TANGERINE'],
        'Zeus_Only':      UNED_COLORS['BLUE'],
        'Full_Optimized': UNED_COLORS['UNED'],
    }
    mode_labels = {
        'Control':        'Control (línea base)',
        'Zeus_Only':      'Zeus Only (power capping)',
        'Full_Optimized': 'Full Optimized (Zeus + EES)',
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, mode in enumerate(modes):
        mode_data = co2_data[co2_data['Mode'] == mode]
        values = [
            mode_data[mode_data['Model'] == m]['CO2g'].values[0]
            if len(mode_data[mode_data['Model'] == m]) > 0 else 0
            for m in models
        ]
        ax.bar(x + (i - 1) * width, values, width,
               label=mode_labels[mode],
               color=mode_colors[mode],
               edgecolor='black', linewidth=0.6)

    ax.set_xlabel('Modelo', fontsize=18, fontweight='bold')
    ax.set_ylabel('Emisiones Acumuladas (g CO₂eq)', fontsize=18, fontweight='bold')
    ax.set_title('Huella de Carbono Acumulada por Modelo y Modo de Optimización\n'
                 '(factor de emisión: España, 0,145 kgCO₂/kWh, Ember 2025)',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend(title='Modo de Optimización', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_5_co2_footprint.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_5_co2_footprint.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 5: CO₂ Footprint")


def plot_accuracy_vs_energy(df):
    """
    Gráfica 6: Trade-off energía-exactitud (Control vs. Full Optimized).
    
    Visualiza ganar-ganar (negativo en Y = mejora de accuracy) versus
    casos de penalización energético-computacional.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    # Ahorro promediado por configuración (coherente con la gráfica de impacto y con
    # las tablas); pérdida de exactitud en PUNTOS PORCENTUALES, que es la unidad que
    # emplea el texto de la memoria.
    result = []
    for model in sorted(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        savings_per_cfg, loss_per_cfg = [], []

        for batch in sorted(model_df['Batch'].unique()):
            cfg = model_df[model_df['Batch'] == batch]
            control = cfg[cfg['Mode'] == 'Control']
            full    = cfg[cfg['Mode'] == 'Full_Optimized']
            if control.empty or full.empty:
                continue
            ctrl_energy = control['EnergyJ'].mean()
            if ctrl_energy <= 0:
                continue
            savings_per_cfg.append(((ctrl_energy - full['EnergyJ'].mean()) / ctrl_energy) * 100)
            # Valores negativos = Full_Optimized alcanzó mayor exactitud (ganar-ganar)
            loss_per_cfg.append((control['Accuracy'].mean() - full['Accuracy'].mean()) * 100)

        if savings_per_cfg:
            result.append({
                'Modelo':              model,
                'AhorroEnergetico_%':  float(np.mean(savings_per_cfg)),
                'PerdidaAccuracy_pp':  float(np.mean(loss_per_cfg)),
            })

    result_df = pd.DataFrame(result)

    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = ax.scatter(
        result_df['AhorroEnergetico_%'],
        result_df['PerdidaAccuracy_pp'],
        s=200, alpha=0.85,
        c=[MODEL_COLORS.get(m, 'gray') for m in result_df['Modelo']],
        edgecolor='black', linewidth=1.2, zorder=3
    )

    for idx, row in result_df.iterrows():
        ax.annotate(
            row['Modelo'],
            (row['AhorroEnergetico_%'], row['PerdidaAccuracy_pp']),
            textcoords="offset points", xytext=(8, 4),
            fontsize=16, fontweight='bold'
        )

    ax.axhline(y=0, color=UNED_COLORS['STRAWBERRY'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Sin pérdida de exactitud (y = 0)')
    # Sombrear zona beneficiosa (ahorro + sin pérdida)
    xlim = ax.get_xlim()
    ax.axhspan(-10, 0, alpha=0.05, color=UNED_COLORS['APPLE'],
               label='Zona ganar-ganar (ahorro + mejora accuracy)')

    ax.set_xlabel('Ahorro Energético (%) respecto a Control', fontsize=18, fontweight='bold')
    ax.set_ylabel('Variación de Exactitud (puntos porcentuales) [negativo = mejora]',
                  fontsize=18, fontweight='bold')
    ax.set_title('Trade-off Energía vs. Exactitud: Control frente a Full Optimized',
                 fontsize=19, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_6_accuracy_vs_energy.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_6_accuracy_vs_energy.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 6: Accuracy vs. Energy Loss")


def plot_energy_heatmap(df):
    """
    Gráfica 7: Mapa de calor J/muestra (Modelo × Batch Size).
    
    Heatmap en escala de rojos (fresa UNED) mostrando la intensidad energética
    para cada combinación de arquitectura y batch size. Se usa rojo y no verde
    porque la magnitud representada es consumo: el extremo intenso de la escala
    corresponde al peor caso energético.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    control_data = df[df['Mode'] == 'Control'].copy()
    pivot_df = control_data.pivot_table(
        index='Model', columns='Batch', values='JSample', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_df, annot=True, fmt=".4f",
        cmap=sns.light_palette(UNED_COLORS['STRAWBERRY'], as_cmap=True),
        cbar_kws={'label': 'Joules por Muestra (J/muestra)'},
        linewidths=0.5, linecolor='white',
        ax=ax
    )
    ax.set_title('Mapa de Calor: Intensidad Energética por Arquitectura y Tamaño de Batch\n'
                 '(Modo Control — valores en J/muestra)',
                 fontsize=18, fontweight='bold')
    ax.set_xlabel('Tamaño de Batch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Arquitectura', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_7_energy_heatmap.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_7_energy_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 7: Heatmap de Energía")


def plot_layer_energy_resnet18():
    """
    Gráfica 8: Perfilado de cómputo por capa en ResNet18.
    
    Usa el archivo layer_profile.csv del directorio bs128 de ResNet18.
    Visualiza las 15 capas con mayor tiempo de cómputo.
    
    Nota: Si el archivo no existe, emite advertencia y continúa ejecución.
    """
    # El naming de logs incluye modo y repetición (…_control_rep1). Se prefiere la
    # primera repetición en modo Control; el patrón sin sufijo se mantiene como
    # respaldo para datasets con el naming antiguo.
    patterns = [
        "logs/CIFAR10_ResNet18_cuda0_bs128_fp16_control_rep1/*_layer_profile.csv",
        "logs/CIFAR10_ResNet18_cuda0_bs128_fp16_control_rep*/*_layer_profile.csv",
        "logs/CIFAR10_ResNet18_cuda0_bs128_fp16/*_layer_profile.csv",
    ]
    csv_files = []
    for pattern in patterns:
        csv_files = sorted(glob.glob(str(BASE_DIR / pattern)))
        if csv_files:
            break

    if not csv_files:
        print("⚠️  No se encontró layer_profile.csv para ResNet18/bs128. Omitiendo gráfica 8.")
        return

    layer_file = Path(csv_files[0])
    layer_df   = pd.read_csv(layer_file)

    required_cols = {'layer', 'total_ms', 'time_fraction'}
    if not required_cols.issubset(layer_df.columns):
        print(f"⚠️  Columnas inesperadas en layer_profile: {layer_df.columns.tolist()}")
        return

    top_layers = layer_df.nlargest(15, 'total_ms').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.light_palette(UNED_COLORS['BLUE'], n_colors=len(top_layers))
    bars = ax.barh(top_layers['layer'], top_layers['total_ms'],
                   color=list(reversed(colors)), edgecolor='black', linewidth=0.5)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        frac  = top_layers.iloc[i]['time_fraction'] * 100
        ax.text(width * 1.005, bar.get_y() + bar.get_height() / 2,
                f'{frac:.1f}% ({width:.0f} ms)',
                ha='left', va='center', fontsize=15, fontweight='bold')

    ax.set_xlabel('Tiempo Total de Cómputo (ms)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Capa del Modelo', fontsize=18, fontweight='bold')
    ax.set_title('Perfilado de Cómputo por Capa: ResNet18 — Top 15 capas (batch=128)',
                 fontsize=18, fontweight='bold')
    ax.set_xlim(0, top_layers['total_ms'].max() * 1.22)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_8_layer_profile_resnet18.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_8_layer_profile_resnet18.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 8: Perfilado por Capa (ResNet18)")


def plot_co2_regional_heatmap(df):
    """
    Gráfica 9: Mapa de calor CO₂ por arquitectura × región geográfica.
    
    Visualiza cómo los factores de emisión regional (Ember 2025) impactan
    la huella de carbono de cada modelo, mostrando variación de hasta 24×
    (Noruega vs India).
    
    Usa batch_size=256 en modo Control como línea base.
    
    Args:
        df (pd.DataFrame): DataFrame con datos del benchmark.
    """
    control_bs256 = df[(df['Mode'] == 'Control') & (df['Batch'] == 256)].copy()
    models = sorted(control_bs256['Model'].unique())

    # Energía media por modelo sobre las cinco repeticiones (coherente con la tabla
    # de CO₂ regional; antes se sobrescribía con la última repetición).
    mean_energy = control_bs256.groupby('Model')['EnergyJ'].mean()

    # Construir matriz: filas=modelos, columnas=regiones
    matrix = pd.DataFrame(index=models, columns=list(CO2_FACTORS.keys()), dtype=float)
    for model in models:
        energy_kwh = mean_energy[model] / 3_600_000  # J → kWh
        for region, factor in CO2_FACTORS.items():
            matrix.loc[model, region] = round(energy_kwh * factor * 1000, 2)

    matrix = matrix.astype(float)

    # Compute real min/max ignoring NaN; enforce vmin >= 0.1 to keep LogNorm valid
    flat_vals = matrix.values.flatten()
    flat_vals = flat_vals[~np.isnan(flat_vals) & (flat_vals > 0)]
    vmin = max(float(flat_vals.min()) * 0.9, 0.1)
    vmax = float(flat_vals.max()) * 1.1

    fig, ax = plt.subplots(figsize=(16, 7))

    # Gradiente de naranja (UNED tangerine) con escala logarítmica
    cmap = sns.light_palette(UNED_COLORS['TANGERINE'], as_cmap=True)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Anotaciones con 2 decimales para preservar valores pequeños (ej. 0.37 g)
    sns.heatmap(
        matrix,
        annot=True, fmt=".2f",
        cmap=cmap,
        norm=norm,
        cbar_kws={'label': 'Emisiones CO₂ (g) por entrenamiento completo (escala log)'},
        linewidths=0.4, linecolor='white',
        ax=ax
    )

    # Formatear colorbar con ticks en órdenes de magnitud legibles
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.1, 0.3, 1, 3, 10, 30, 100])
    cbar.set_ticklabels(['0.1', '0.3', '1', '3', '10', '30', '100'])
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Emisiones CO₂ (g) por entrenamiento completo (escala log)',
                   fontsize=18)

    ax.set_title('Impacto Geográfico del Entrenamiento: Emisiones CO₂ (gCO₂eq)\n'
                 'por Arquitectura × País (batch=256, modo Control, 50 épocas)',
                 fontsize=20, fontweight='bold')
    ax.set_xlabel('Región Geográfica (factor de emisión Ember 2025)', fontsize=19,
                  fontweight='bold')
    ax.set_ylabel('Arquitectura', fontsize=19, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS / "grafica_9_co2_heatmap_geografico.pdf",
                bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PLOTS / "grafica_9_co2_heatmap_geografico.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("🎨 Gráfica 9: Heatmap CO₂ Regional (LogNorm)")


# ============================================================================
# FUNCIÓN PRINCIPAL Y ORQUESTACIÓN
# ============================================================================

def main():
    """
    Orquesta la ejecución completa del análisis del benchmark.
    
    Pasos:
    1. Validación de directorios de salida
    2. Carga de datos del archivo maestro JSON
    3. Generación de 10 tablas CSV
    4. Generación de 9 gráficas PDF
    5. Reporte de conclusión
    
    Raises:
        FileNotFoundError: Si falta el archivo de datos maestro
        Exception: Otros errores fatales durante análisis
    """
    print("\n" + "=" * 80)
    print("  SUITE INTEGRAL DE ANÁLISIS DEL BENCHMARK")
    print("  Framework Green AI para Eficiencia Energética en Deep Learning")
    print("  Refactorización: 2026-04-15")
    print("=" * 80)
    print(f"\n⏱️  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Validación
        print("🔍 Validando entorno...")
        create_output_directories()
        validate_output_paths()
        print("   ✓ Directorios de salida creados/validados\n")

        # Carga de datos
        print("📊 Cargando datos del benchmark...")
        df = load_benchmark_data()
        print(f"   ✓ {len(df)} registros cargados exitosamente")
        print(f"   ✓ Modelos: {sorted(df['Model'].unique())}")
        print(f"   ✓ Modos: {sorted(df['Mode'].unique())}")
        print(f"   ✓ Batch sizes: {sorted(df['Batch'].unique())}\n")

        # Tablas CSV
        print("📋 GENERANDO TABLAS CSV (10 análisis comparativos)...")
        generate_tabla_zeus_impact(df)
        generate_tabla_ees_impact(df)
        generate_tabla_ranking_global(df)
        generate_tabla_batch_sweep(df)
        generate_tabla_by_architecture(df)
        generate_tabla_co2_normalized(df)
        generate_tabla_estadisticas(df)
        generate_tabla_metadata()
        generate_tabla_significancia(df)
        generate_tabla_co2_regional(df)

        # Gráficas PDF
        print("\n🎨 GENERANDO GRÁFICAS PDF VECTORIALES (9 visualizaciones)...")
        plot_pareto(df)
        plot_energy_intensity(df)
        plot_batch_scaling(df)
        plot_optimization_impact(df)
        plot_co2_footprint(df)
        plot_accuracy_vs_energy(df)
        plot_energy_heatmap(df)
        plot_layer_energy_resnet18()
        plot_co2_regional_heatmap(df)

        # Conclusión
        print("\n" + "=" * 80)
        print("✨ ¡ANÁLISIS COMPLETADO CON ÉXITO!")
        print("=" * 80)
        print(f"\n📁 Resultados generados:")
        print(f"   📊 CSVs (10 tablas):    {OUTPUT_CSV}")
        print(f"   📈 PDFs (9 gráficas):   {OUTPUT_PLOTS}")
        print(f"   📄 Metadatos:           {OUTPUT_METADATA}")
        print(f"\n⏱️  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Ejecute el benchmark antes de este script.")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
