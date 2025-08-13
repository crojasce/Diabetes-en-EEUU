#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_pca_mca.py
-------------
Realiza PCA (numéricas) y MCA (categóricas) sobre un dataset,
selecciona componentes/dimensiones hasta superar un umbral de varianza/inercia acumulada,
y guarda un nuevo dataset con las PCs y las Dims concatenadas.

Uso:
    python 01_pca_mca.py \
        --input_csv /ruta/diabetic_data.csv \
        --ids_mapping /ruta/IDS_mapping.csv \
        --output_csv /ruta/dataset_pca_mca.csv \
        --threshold 0.80

Notas:
- 'ids_mapping' es opcional (por si luego quieres hacer merges/enriquecimiento).
- Se excluyen columnas identificadoras por heurística: nombres con 'id' o unicidad alta.
- PCA estandariza las numéricas con StandardScaler.
- MCA requiere la librería 'prince'.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from prince import MCA
except Exception as e:
    raise ImportError(
        "No se pudo importar 'prince'. Instala con: pip install prince\n"
        f"Detalle: {e}"
    )

# --------------------------
# Utilidades
# --------------------------

def is_identifier(colname: str) -> bool:
    """Heurística simple para detectar columnas identificadoras por nombre."""
    name = str(colname).lower()
    return any(tok in name for tok in ["id", "identifier", "uuid", "nbr", "key"])

def drop_identifier_like_columns(df: pd.DataFrame, id_name_check=True, high_uniqueness_ratio=0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas que parecen identificadores:
    - Por nombre (contiene 'id', 'uuid', 'nbr', etc.)
    - Por unicidad (nunique ~ número de filas)
    """
    cols_to_drop = set()

    nrows = len(df)
    if nrows == 0:
        return df.copy(), []

    if id_name_check:
        for c in df.columns:
            if is_identifier(c):
                cols_to_drop.add(c)

    # alta unicidad
    for c in df.columns:
        try:
            unique_ratio = df[c].nunique(dropna=True) / max(1, nrows)
            if unique_ratio >= high_uniqueness_ratio:
                cols_to_drop.add(c)
        except Exception:
            # Si algo raro pasa al calcular nunique, no descartamos por este criterio
            pass

    cleaned = df.drop(columns=list(cols_to_drop), errors="ignore")
    return cleaned, sorted(list(cols_to_drop))


def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa dataframes numéricos y categóricos según dtype."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    categorical_df = df.select_dtypes(exclude=["number"]).copy()
    return numeric_df, categorical_df


def fit_pca(numeric_df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Estandariza y ajusta PCA. Devuelve:
    - PCs (hasta alcanzar el umbral)
    - Objeto PCA ajustado
    - Varianza acumulada
    """
    if numeric_df.shape[1] == 0:
        return pd.DataFrame(index=numeric_df.index), None, np.array([])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_num = scaler.fit_transform(numeric_df.fillna(numeric_df.mean(numeric_only=True)))

    pca = PCA()
    X_pca = pca.fit_transform(X_num)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    # cantidad mínima de componentes para llegar al umbral
    n_components = int(np.searchsorted(cum_var, threshold) + 1)

    pcs = pd.DataFrame(
        X_pca[:, :n_components],
        index=numeric_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pcs, pca, cum_var


def fit_mca(categorical_df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, MCA, np.ndarray]:
    """
    Ajusta MCA sobre categóricas. Devuelve:
    - Coordenadas MCA (hasta alcanzar el umbral de inercia acumulada)
    - Objeto MCA ajustado
    - Inercia acumulada
    """
    if categorical_df.shape[1] == 0:
        return pd.DataFrame(index=categorical_df.index), None, np.array([])

    cat = categorical_df.fillna("missing").astype(str)

    # prince.MCA calcula explained_inertia_ tras fit
    mca = MCA()  # parámetros por defecto
    mca = mca.fit(cat)
    coords = mca.transform(cat)

    # explained_inertia_ es una serie con la inercia por dimensión
    inertia = np.array(mca.explained_inertia_)
    cum_inertia = np.cumsum(inertia)

    n_dims = int(np.searchsorted(cum_inertia, threshold) + 1)
    dims = coords.iloc[:, :n_dims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(n_dims)]

    return dims, mca, cum_inertia


def main(args):
    # --------------------------
    # Carga
    # --------------------------
    if not os.path.exists(args.input_csv):
        print(f"[ERROR] No existe el archivo de entrada: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input_csv)

    # (Opcional) cargar mapping si se requiere más tarde
    if args.ids_mapping and os.path.exists(args.ids_mapping):
        ids_map = pd.read_csv(args.ids_mapping)
        # No se usa en este script, pero queda cargado si quieres incorporarlo después.
    else:
        ids_map = None

    # --------------------------
    # Limpieza: remover columnas identificadoras
    # --------------------------
    df_clean, dropped_ids = drop_identifier_like_columns(df, id_name_check=True, high_uniqueness_ratio=0.95)
    if dropped_ids:
        print(f"[INFO] Columnas descartadas por parecer identificadores: {dropped_ids}")

    # --------------------------
    # Separación por tipo
    # --------------------------
    num_df, cat_df = split_numeric_categorical(df_clean)
    print(f"[INFO] Numéricas: {num_df.shape[1]} columnas | Categóricas: {cat_df.shape[1]} columnas")

    # --------------------------
    # PCA (numéricas)
    # --------------------------
    pcs, pca_model, cum_var = fit_pca(num_df, threshold=args.threshold)
    if pca_model is not None:
        print(f"[INFO] PCA -> componentes retenidas: {pcs.shape[1]}")
        print(f"[INFO] Varianza acumulada (primeras 10): {np.round(cum_var[:10], 4)}")
    else:
        print("[INFO] PCA omitido: no hay columnas numéricas.")

    # --------------------------
    # MCA (categóricas)
    # --------------------------
    dims, mca_model, cum_inertia = fit_mca(cat_df, threshold=args.threshold)
    if mca_model is not None:
        print(f"[INFO] MCA -> dimensiones retenidas: {dims.shape[1]}")
        print(f"[INFO] Inercia acumulada (primeras 10): {np.round(cum_inertia[:10], 4)}")
    else:
        print("[INFO] MCA omitido: no hay columnas categóricas.")

    # --------------------------
    # Concatenación y guardado
    # --------------------------
    if pcs.shape[0] == 0 and dims.shape[0] == 0:
        print("[ERROR] No hay datos transformados para guardar.", file=sys.stderr)
        sys.exit(2)

    out_df = pd.concat([pcs, dims], axis=1)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[OK] Dataset final guardado en: {args.output_csv}")
    print(f"[OK] Forma del dataset: {out_df.shape[0]} filas x {out_df.shape[1]} columnas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA (numéricas) + MCA (categóricas) y concatenación")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta a diabetic_data.csv")
    parser.add_argument("--ids_mapping", type=str, default="", help="Ruta a IDS_mapping.csv (opcional)")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta de salida para el CSV resultante")
    parser.add_argument("--threshold", type=float, default=0.80, help="Umbral de varianza/inercia acumulada (default 0.80)")
    args = parser.parse_args()
    main(args)
