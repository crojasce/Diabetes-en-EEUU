
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Taller 1 - Tarea 1
PCA (numéricas) + MCA (categóricas) con umbral de varianza acumulada (por defecto 0.90).
- MCA se implementa como One-Hot Encoding + TruncatedSVD (equivalente a PCA para matrices dispersas).
- Los identificadores 'encounter_id' y 'patient_nbr' se excluyen. 
- Columnas *_id se tratan como categóricas, aunque sean int.

Uso:
    python src/01_pca_mca.py \
        --data data/diabetic_data.csv \
        --out outputs/dataset_pca_mca.csv \
        --var 0.90
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from scipy import sparse


# --------------------------
# Utilidades
# --------------------------
EXCLUDE_ID_COLS = ["encounter_id", "patient_nbr"]

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Detecta columnas numéricas y categóricas.
    Trata como categóricas las de tipo 'object' y también las que terminan en '_id' (salvo exclusiones)."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Forzar *_id como categóricas (menos las excluidas)
    for c in df.columns:
        if c.endswith("_id") and c not in EXCLUDE_ID_COLS:
            if c in num_cols:
                num_cols.remove(c)
            if c not in cat_cols:
                cat_cols.append(c)

    # Excluir identificadores globales
    num_cols = [c for c in num_cols if c not in EXCLUDE_ID_COLS]
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_ID_COLS]

    return num_cols, cat_cols


def fit_pca_numeric(X_num: pd.DataFrame, var_target: float, random_state: int = 42) -> Tuple[np.ndarray, PCA]:
    """Imputa (mediana), estandariza y aplica PCA preservando var_target de varianza."""
    if X_num.shape[1] == 0:
        return np.zeros((len(X_num), 0)), None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_num_imputed = imputer.fit_transform(X_num)
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    # PCA puede recibir un float (0-1) como n_components para varianza objetivo.
    pca = PCA(n_components=var_target, svd_solver="full", random_state=random_state)
    Z_num = pca.fit_transform(X_num_scaled)
    return Z_num, pca


def choose_svd_components(X_sparse: sparse.spmatrix, var_target: float,
                          start: int = 50, step: int = 50,
                          random_state: int = 42) -> Tuple[np.ndarray, TruncatedSVD, int]:
    """Ajusta TruncatedSVD incrementando componentes hasta alcanzar var_target.
    Devuelve la proyección, el modelo entrenado y el k mínimo."""
    n_max = min(X_sparse.shape) - 1
    if n_max <= 0:
        return np.zeros((X_sparse.shape[0], 0)), None, 0

    n = min(start, n_max)
    best_svd, best_cum, best_k = None, None, 0

    while True:
        svd = TruncatedSVD(n_components=n, random_state=random_state)
        svd.fit(X_sparse)
        cum = svd.explained_variance_ratio_.cumsum()
        k = int(np.searchsorted(cum, var_target) + 1)

        best_svd, best_cum, best_k = svd, cum, k
        if cum[-1] >= var_target or n == n_max:
            break
        n = min(n + step, n_max)

    Z_cat = best_svd.transform(X_sparse)[:, :best_k]
    return Z_cat, best_svd, best_k


def fit_mca_categorical(df: pd.DataFrame, cat_cols: List[str],
                        var_target: float, random_state: int = 42) -> Tuple[np.ndarray, TruncatedSVD, int, List[str]]:
    """One-Hot (sparse) + TruncatedSVD (MCA). Reemplaza '?' por NaN y usa 'Unknown' como categoría."""
    if len(cat_cols) == 0:
        return np.zeros((len(df), 0)), None, 0, []

    X_cat_raw = df[cat_cols].copy()
    X_cat_raw = X_cat_raw.replace("?", np.nan)

    # OneHotEncoder: crea 'Unknown' para NaN, ignora categorías no vistas
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    X_cat = ohe.fit_transform(X_cat_raw.fillna("Unknown"))

    Z_cat, svd, k = choose_svd_components(X_cat, var_target=var_target, random_state=random_state)

    # nombres de dimensiones MCA
    mca_cols = [f"MCA_{i+1}" for i in range(k)]
    return Z_cat, svd, k, mca_cols


def build_embeddings(df: pd.DataFrame, var_target: float = 0.90, random_state: int = 42) -> pd.DataFrame:
    """Construye el dataset final con componentes PCA + MCA."""
    num_cols, cat_cols = detect_columns(df)

    Z_num, pca = fit_pca_numeric(df[num_cols], var_target, random_state)
    pca_cols = [f"PCA_{i+1}" for i in range(Z_num.shape[1])]

    Z_cat, svd, k_cat, mca_cols = fit_mca_categorical(df, cat_cols, var_target, random_state)

    # Concatenar (mantenemos el mismo índice que df)
    Z = np.hstack([Z_num, Z_cat]) if Z_cat.size and Z_num.size else (Z_num if Z_cat.size == 0 else Z_cat)
    out_cols = pca_cols + mca_cols
    emb = pd.DataFrame(Z, index=df.index, columns=out_cols)

    return emb


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="PCA (num) + MCA (cat) con varianza acumulada objetivo.")
    parser.add_argument("--data", required=True, help="Ruta a diabetic_data.csv")
    parser.add_argument("--out", required=True, help="Ruta del CSV de salida con las componentes concatenadas")
    parser.add_argument("--var", type=float, default=0.90, help="Varianza acumulada objetivo (0-1). Default=0.90")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.data)
    # Excluir IDs globales del conjunto final (pero conservamos el índice/orden)
    df_proc = df.drop(columns=[c for c in EXCLUDE_ID_COLS if c in df.columns], errors="ignore")

    emb = build_embeddings(df_proc, var_target=args.var, random_state=42)
    emb.to_csv(args.out, index=False)

    # Log rápido
    n_pca = sum(col.startswith("PCA_") for col in emb.columns)
    n_mca = sum(col.startswith("MCA_") for col in emb.columns)
    print(f"[OK] Componentes PCA: {n_pca} | Dimensiones MCA: {n_mca} | Total: {emb.shape[1]}")
    print(f"Archivo guardado en: {args.out}")


if __name__ == "__main__":
    main()
