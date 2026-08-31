#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_utils.py

Módulo centralizado de configuración y utilidades reutilizables para análisis de benchmarks.

Contiene:
- Configuraciones de rutas (INPUT, OUTPUT)
- Paleta de colores UNED institucional
- Factores de emisión CO₂ por región geográfica
- Funciones estadísticas comunes (intervalo de confianza, frontera de Pareto)

Este módulo permite mantener consistencia en toda la suite de análisis y evitar
redundancia en la configuración entre scripts múltiples.

Autor: Refactorización Senior Software Architect
Fecha: 2026-04-15
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# El grader del framework es la única fuente de verdad para la escala A++–F.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from moniaenergy.metrics.green_grader import compute_grade, calibrate_reference  # noqa: E402

# ============================================================================
# CONFIGURACIÓN CENTRALIZADA DE RUTAS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
"""Directorio raíz del proyecto."""

INPUT_FILE = BASE_DIR / "benchmark_master_dataset.json"
"""Archivo de datos maestro: resultados agregados del benchmark."""

OUTPUT_CSV = BASE_DIR / "results" / "csv"
"""Directorio de salida para tablas CSV."""

OUTPUT_PLOTS = Path(
    os.environ.get("MONIAENERGY_FIGURES_DIR", BASE_DIR / "results" / "figuras")
)
"""Directorio de salida para gráficas PDF.

Por defecto ``results/figuras``. Se puede redirigir a la carpeta de figuras de
una memoria LaTeX externa mediante la variable de entorno
``MONIAENERGY_FIGURES_DIR``.
"""

OUTPUT_METADATA = BASE_DIR / "results" / "metadata"
"""Directorio de salida para metadatos y archivos de contexto."""

OUTPUT_PLOTS_PNG = BASE_DIR / "results" / "plots"
"""Directorio de salida para gráficas PNG (uso en presentaciones)."""


def create_output_directories():
    """
    Crea todos los directorios de salida si no existen.
    
    Raises:
        OSError: Si falla la creación de directorios.
    """
    for directory in [OUTPUT_CSV, OUTPUT_PLOTS, OUTPUT_METADATA, OUTPUT_PLOTS_PNG]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PALETA DE COLORES INSTITUCIONAL (UNED)
# ============================================================================

UNED_COLORS = {
    'UNED':              '#00533E',   # Verde corporativo UNED
    'UNED_MEDIUM':       '#427562',   # Verde oscuro intermedio
    'UNED_LIGHT':        '#A8C8B8',   # Verde claro
    'APPLE':             '#749F4C',   # Verde manzana
    'BLUE':              '#5C6EB1',   # Azul institucional
    'BLUE_MEDIUM':       '#7683BD',   # Azul intermedio
    'TANGERINE':         '#D76F47',   # Naranja tangerina
    'STRAWBERRY':        '#DA5268',   # Fresa
    'RASPBERRY':         '#90214A',   # Frambuesa (más oscuro)
}
"""Paleta de colores oficiales UNED para visualizaciones consistentes."""

MODEL_COLORS = {
    'VGG19':              UNED_COLORS['BLUE'],
    'ResNet50':           UNED_COLORS['TANGERINE'],
    'MobileNetV2':        UNED_COLORS['APPLE'],
    'DenseNet121':        UNED_COLORS['STRAWBERRY'],
    'EfficientNetB0':     UNED_COLORS['RASPBERRY'],
    'ResNet18':           UNED_COLORS['UNED'],
    'ViT':                UNED_COLORS['BLUE_MEDIUM'],
}
"""Mapeo de colores por arquitectura de red neuronal (coherente en todas las gráficas)."""

# ============================================================================
# FACTORES DE EMISIÓN CO₂ Y CONTEXTO GEOGRÁFICO
# ============================================================================

CO2_FACTORS = {
    'España':           0.145,
    'Francia':          0.065,
    'Alemania':         0.380,
    'EE. UU.':          0.384,
    'Noruega':          0.029,
    'Reino Unido':      0.185,
    'China':            0.560,
    'India':            0.708,
    'Brasil':           0.103,
    'Australia':        0.480,
    'Canadá':           0.120,
    'Japón':            0.485,
}
"""
Factores de emisión de CO₂ por región (kgCO₂/kWh).

Fuente: Ember 2025. Coherentes con el factor de España (145 gCO₂/kWh) que emplea
el módulo de estimación de carbono del framework, de modo que las tablas, las figuras
y el texto usan un único conjunto de factores.

Rango: Noruega (0.029) - India (0.708) — variación de ~24×.
"""


# ============================================================================
# FUNCIONES ESTADÍSTICAS COMUNES
# ============================================================================

def calculate_ci(data, confidence=0.95):
    """
    Calcula intervalo de confianza (IC) usando t-distribution.
    
    En conjuntos pequeños (n < 30), la t-distribution es más precisa que la
    distribución normal.
    
    Args:
        data (array-like): Datos numéricos.
        confidence (float): Nivel de confianza (defecto: 0.95 = 95%).
    
    Returns:
        tuple: (límite_inferior, límite_superior) del IC.
        En caso de datos insuficientes (n < 2), retorna (nan, nan).
    
    Examples:
        >>> ci_low, ci_high = calculate_ci([1, 2, 3, 4, 5])
        >>> print(f"IC95%: [{ci_low:.2f}, {ci_high:.2f}]")
    """
    if len(data) < 2:
        return np.nan, np.nan
    
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean - margin, mean + margin


def pareto_frontier(df_in, x_col='EnergyJ', y_col='Accuracy'):
    """
    Calcula la frontera de Pareto-óptimos para minimizar/maximizar dos objetivos.
    
    En el contexto de eficiencia energética:
    - **Minimizar**: Energía (x_col) — menor consumo es mejor.
    - **Maximizar**: Exactitud (y_col) — mayor precisión es mejor.
    
    Retorna los puntos no dominados (no existe otro punto que sea mejor en ambos
    objetivos simultáneamente).
    
    Args:
        df_in (pd.DataFrame): DataFrame con mínimo dos columnas numéricas.
        x_col (str): Nombre de columna para eje X (objetivo de minimización).
        y_col (str): Nombre de columna para eje Y (objetivo de maximización).
    
    Returns:
        pd.DataFrame: Subset de df_in con puntos Pareto-óptimos, ordenados
                     crecientemente por x_col.
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'energia': [100, 90, 80, 95],
        ...     'accuracy': [0.90, 0.92, 0.88, 0.91]
        ... })
        >>> pareto = pareto_frontier(df, 'energia', 'accuracy')
    """
    df_sorted = df_in.sort_values(x_col).reset_index(drop=True)
    pareto = []
    max_y = -np.inf
    
    for _, row in df_sorted.iterrows():
        if row[y_col] > max_y:
            pareto.append(row)
            max_y = row[y_col]
    
    return pd.DataFrame(pareto)


def load_benchmark_data():
    """
    Carga y estructura los datos agregados del benchmark desde JSON.
    
    La estructura esperada del JSON es:
    ```json
    [
        {
            "model_name": "ResNet18",
            "runs": [
                {
                    "batch_size": 32,
                    "mode": "Control",
                    "optimization_mode": "Control",
                    "flags": {"power_optimize": false, "early_stopping": false},
                    "summary": {
                        "energy_j_total": 45000,
                        "energy_j_mean_epoch": 900,
                        "j_sample_mean": 0.00045,
                        "final_accuracy": 0.92,
                        "co2_g_total": 3.2,
                        "energy_grade": "A+"
                    },
                    "epochs_history": [...]
                },
                ...
            ]
        },
        ...
    ]
    ```
    
    Returns:
        pd.DataFrame: DataFrame con columnas normalizadas:
        - Model, Batch, Mode, PowerOpt, EarlyStopping
        - EnergyJ, EnergyJMean, JSample, Accuracy, CO2g, Grade
        - AvgEDP, AvgGpuEnergyWh, AvgCpuEnergyWh, AccuracyMax, AccuracyGrowth
    
    Raises:
        FileNotFoundError: Si benchmark_master_dataset.json no existe.
        json.JSONDecodeError: Si el archivo JSON es inválido.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Archivo de datos maestro no encontrado: {INPUT_FILE}\n"
            f"Ejecute primero el benchmark para generar este archivo."
        )
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened = []
    for model_entry in data:
        model_name = model_entry['model_name']
        for run in model_entry['runs']:
            row = {
                'Model':           model_name,
                'Batch':           run.get('batch_size', 0),
                'Rep':             run.get('rep', 1),
                'Mode':            run.get('mode', run.get('optimization_mode', 'Unknown')),
                'PowerOpt':        run.get('flags', {}).get('power_optimize', False),
                'EarlyStopping':   run.get('flags', {}).get('early_stopping', False),
                'EpochsCompleted': run.get('epochs_completed', 0),
            }
            
            # Resumen del run
            if run.get('summary'):
                row.update({
                    'EnergyJ':        run['summary'].get('energy_j_total'),
                    'EnergyJMean':    run['summary'].get('energy_j_mean_epoch'),
                    'JSample':        run['summary'].get('j_sample_mean'),
                    'JSampleStd':     run['summary'].get('j_sample_std', 0.0),
                    'Accuracy':       run['summary'].get('final_accuracy'),
                    'CO2g':           run['summary'].get('co2_g_total'),
                    'Grade':          run['summary'].get('energy_grade'),
                })
            
            # Histórico por época
            if run.get('epochs_history'):
                epochs_data = run['epochs_history']
                row['AvgEDP'] = np.mean([e.get('edp', 0) for e in epochs_data])
                row['AvgGpuEnergyWh'] = np.mean([e.get('gpu_energy_wh', 0) for e in epochs_data])
                row['AvgCpuEnergyWh'] = np.mean([e.get('cpu_energy_wh', 0) for e in epochs_data])
                row['AccuracyMax'] = max([e.get('accuracy', 0) for e in epochs_data])
                
                if epochs_data:
                    row['AccuracyGrowth'] = row.get('Accuracy', 0) - epochs_data[0].get('accuracy', 0)
            
            flattened.append(row)
    
    df = pd.DataFrame(flattened)
    
    # Validación básica
    if df.empty:
        raise ValueError("El archivo JSON no contiene datos válidos.")
    
    # --- CÁLCULO POST-HOC DEL GREEN AI GRADE ---
    # Se delega en el propio módulo del framework (moniaenergy.metrics.green_grader)
    # en lugar de reimplementar la escala aquí: así la columna "Grade" de la memoria
    # usa exactamente los umbrales documentados en la Sección "Green AI Grade" y
    # cualquier cambio en el módulo se propaga al análisis sin divergencias.
    def parse_params(p_str):
        if not p_str: return 10
        p_str = str(p_str).upper().replace('M', '').replace('K', '')
        try:
            return int(float(p_str) * 1000000)
        except ValueError:
            return 10

    info = get_model_info()
    df['N_params'] = df['Model'].apply(lambda m: parse_params(info.get(m, {}).get('parametros', '10')))

    # eff_score = acc * log10(max(N_params, 10)) / E_total
    valid_mask = df['EnergyJ'] > 0
    df['eff_score'] = 0.0
    df.loc[valid_mask, 'eff_score'] = (
        df.loc[valid_mask, 'Accuracy'] *
        df.loc[valid_mask, 'N_params'].apply(lambda p: np.log10(max(p, 10)))
    ) / df.loc[valid_mask, 'EnergyJ']

    # Referencia auto-calibrada: mediana de las puntuaciones del propio benchmark,
    # vía calibrate_reference() del framework. La calificación es por tanto relativa
    # a este banco de pruebas (B = desempeño mediano), no una escala absoluta.
    s_ref = calibrate_reference(df.loc[valid_mask, 'eff_score'].tolist())

    def assign_grade(row):
        result = compute_grade(
            total_energy_j=row['EnergyJ'],
            accuracy=row['Accuracy'],
            parameters=int(row['N_params']),
            reference_score=s_ref,
        )
        return result.grade if result else 'F'

    df['Grade'] = df.apply(assign_grade, axis=1)
    df.attrs['grade_reference_score'] = s_ref

    return df


# ============================================================================
# UTILIDADES GENERALES
# ============================================================================

def validate_output_paths():
    """
    Valida que las rutas de salida sean accesibles y escribibles.
    
    Returns:
        bool: True si todos los directorios son válidos.
    
    Raises:
        PermissionError: Si no hay permisos de escritura.
    """
    for directory in [OUTPUT_CSV, OUTPUT_PLOTS, OUTPUT_METADATA, OUTPUT_PLOTS_PNG]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"No se puede crear directorio: {directory}")
    return True


def get_model_info():
    """
    Retorna información de referencia sobre arquitecturas de redes neuronales.
    
    Returns:
        dict: Mapeo de modelo a atributos (complejidad, tipo, etc.).
    """
    return {
        'VGG19': {
            'tipo': 'Pesada',
            'parametros': '144M',
            'año': 2014,
            'receptivo_global': True,
        },
        'ResNet50': {
            'tipo': 'Pesada',
            'parametros': '26M',
            'año': 2015,
            'skip_connections': True,
        },
        'ResNet18': {
            'tipo': 'Media',
            'parametros': '11M',
            'año': 2015,
            'skip_connections': True,
        },
        'MobileNetV2': {
            'tipo': 'Ligera',
            'parametros': '3.5M',
            'año': 2018,
            'depthwise_separable': True,
        },
        'EfficientNetB0': {
            'tipo': 'Ligera',
            'parametros': '5.3M',
            'año': 2019,
            'compound_scaling': True,
        },
        'DenseNet121': {
            'tipo': 'Pesada',
            'parametros': '7.98M',
            'año': 2016,
            'skip_global': True,
        },
    }


if __name__ == '__main__':
    # Validación básica del módulo
    create_output_directories()
    print("✅ Módulo analysis_utils.py cargado correctamente")
    print(f"📁 Rutas configuradas:")
    print(f"   INPUT:    {INPUT_FILE}")
    print(f"   OUTPUT_CSV: {OUTPUT_CSV}")
    print(f"   OUTPUT_PLOTS: {OUTPUT_PLOTS}")
