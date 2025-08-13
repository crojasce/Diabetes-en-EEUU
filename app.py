# app.py
# ------------------------------------------------------------
# App educativa para PCA (numéricas) + MCA (categóricas)
# Selección de componentes/dimensiones por umbral de varianza/inercia acumulada
# y concatenación de PCs + Dims en un solo dataset.
#
# Requisitos:
#   pip install streamlit pandas numpy scikit-learn prince
#
# Ejecuta:
#   streamlit run app.py
# ------------------------------------------------------------

import os
import io
import numpy as np
import pandas as pd
import streamlit as st

from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Intentamos importar prince.MCA de forma segura
try:
    from prince import MCA
    PRINCE_OK = True
except Exception as e:
    PRINCE_OK = False
    PRINCE_ERR = str(e)

# ============================
# Utilidades
# ============================

def is_identifier(colname: str) -> bool:
    """Heurística simple para detectar columnas identificadoras por nombre."""
    name = str(colname).lower()
    return any(tok in name for tok in ["id", "identifier", "uuid", "nbr", "key", "encounter"])

def drop_identifier_like_columns(
    df: pd.DataFrame,
    id_name_check: bool = True,
    high_uniqueness_ratio: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
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

    for c in df.columns:
        try:
            unique_ratio = df[c].nunique(dropna=True) / max(1, nrows)
            if unique_ratio >= high_uniqueness_ratio:
                cols_to_drop.add(c)
        except Exception:
            pass

    cleaned = df.drop(columns=list(cols_to_drop), errors="ignore")
    return cleaned, sorted(list(cols_to_drop))

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa dataframes numéricos y categóricos según dtype."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    categorical_df = df.select_dtypes(exclude=["number"]).copy()
    return numeric_df, categorical_df

def fit_pca(numeric_df: pd.DataFrame, threshold: float):
    """
    Estandariza y ajusta PCA. Devuelve:
      - PCs retenidas hasta el umbral
      - objeto PCA
      - varianza explicada acumulada
    """
    if numeric_df.shape[1] == 0:
        return pd.DataFrame(index=numeric_df.index), None, np.array([])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_num = scaler.fit_transform(numeric_df.fillna(numeric_df.mean(numeric_only=True)))

    pca = PCA()
    X_pca = pca.fit_transform(X_num)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, threshold) + 1)

    pcs = pd.DataFrame(
        X_pca[:, :n_components],
        index=numeric_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pcs, pca, cum_var

def fit_mca(categorical_df: pd.DataFrame, threshold: float):
    """
    Ajusta MCA sobre categóricas. Devuelve:
      - coordenadas (Dimensiones) retenidas hasta el umbral
      - objeto MCA
      - inercia acumulada
    """
    if categorical_df.shape[1] == 0:
        return pd.DataFrame(index=categorical_df.index), None, np.array([])

    if not PRINCE_OK:
        raise ImportError(
            "La librería 'prince' no está disponible. Instala con: pip install prince\n"
            f"Detalle: {PRINCE_ERR}"
        )

    cat = categorical_df.fillna("missing").astype(str)
    mca = MCA()
    mca = mca.fit(cat)
    coords = mca.transform(cat)

    inertia = np.array(mca.explained_inertia_)  # lista de inercia por dimensión
    cum_inertia = np.cumsum(inertia)
    n_dims = int(np.searchsorted(cum_inertia, threshold) + 1)

    dims = coords.iloc[:, :n_dims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(n_dims)]
    return dims, mca, cum_inertia

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def pca_mca_pipeline(
    df: pd.DataFrame,
    threshold: float = 0.80,
    drop_ids: bool = True,
    high_uniqueness_ratio: float = 0.95
):
    # 1) limpiar IDs
    df_clean, dropped_ids = (drop_identifier_like_columns(df, True, high_uniqueness_ratio)
                             if drop_ids else (df.copy(), []))
    # 2) separar tipos
    num_df, cat_df = split_numeric_categorical(df_clean)
    # 3) PCA
    pcs, pca_model, cum_var = fit_pca(num_df, threshold)
    # 4) MCA
    dims, mca_model, cum_inertia = fit_mca(cat_df, threshold) if cat_df.shape[1] > 0 else (pd.DataFrame(index=df.index), None, np.array([]))
    # 5) Concatenar
    out_df = pd.concat([pcs, dims], axis=1)
    return {
        "df_clean": df_clean,
        "dropped_ids": dropped_ids,
        "num_df": num_df,
        "cat_df": cat_df,
        "pcs": pcs,
        "pca_model": pca_model,
        "cum_var": cum_var,
        "dims": dims,
        "mca_model": mca_model,
        "cum_inertia": cum_inertia,
        "out_df": out_df
    }

# ============================
# UI
# ============================

st.set_page_config(page_title="PCA + MCA | Diabetes", layout="wide")

st.title("PCA + MCA del dataset de Diabetes en EE. UU.")
st.markdown(
    """
**Objetivo:** Aplicar **PCA** sobre las variables **numéricas** y **MCA** sobre las **categóricas**, 
seleccionar las **componentes/dimensiones** que acumulen **≥ un umbral de varianza/inercia** (por defecto 80%) 
y **crear un nuevo dataset** con las **PCs** y las **Dimensiones** concatenadas.
"""
)

with st.expander("🧩 Requisitos e instrucciones rápidas", expanded=False):
    st.markdown(
        """
- Dependencias: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `prince`  
 """
    )
  ```bash
  pip install streamlit pandas numpy scikit-learn prince







