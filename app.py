# app.py
# ------------------------------------------------------------
# PCA (numéricas) + MCA (categóricas) con umbral del 80%
# - Reduce categorías raras/altas cardinalidades a "OTHER" (más rápido y estable)
# - Muestra paso a paso y entrega dataset final concatenado
# ------------------------------------------------------------

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importar prince.MCA de forma segura
try:
    from prince import MCA
    PRINCE_OK = True
    PRINCE_ERR = ""
except Exception as e:
    PRINCE_OK = False
    PRINCE_ERR = str(e)

# ================= Utilidades =================

def is_identifier(colname: str) -> bool:
    name = str(colname).lower()
    return any(tok in name for tok in ["id", "identifier", "uuid", "nbr", "key", "encounter"])

def drop_identifier_like_columns(df: pd.DataFrame, high_uniqueness_ratio: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    cols_to_drop = set()
    nrows = len(df)

    for c in df.columns:
        if is_identifier(c):
            cols_to_drop.add(c)

    if nrows > 0:
        for c in df.columns:
            try:
                if df[c].nunique(dropna=True) / nrows >= high_uniqueness_ratio:
                    cols_to_drop.add(c)
            except Exception:
                pass

    cleaned = df.drop(columns=list(cols_to_drop), errors="ignore")
    return cleaned, sorted(list(cols_to_drop))

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df.select_dtypes(include=["number"]).copy()
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    return num_df, cat_df

def fit_pca(num_df: pd.DataFrame, threshold: float = 0.80):
    if num_df.shape[1] == 0:
        return pd.DataFrame(index=num_df.index), None, np.array([])

    X = num_df.fillna(num_df.mean(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)

    pca = PCA()
    Xp = pca.fit_transform(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cum_var, threshold) + 1)

    pcs = pd.DataFrame(Xp[:, :ncomp], index=num_df.index,
                       columns=[f"PC{i+1}" for i in range(ncomp)])
    return pcs, pca, cum_var

def reduce_categorical_cardinality(cat: pd.DataFrame, max_modalities_per_col: int = 50, rare_min_count: int = 30) -> pd.DataFrame:
    """
    - Convierte a str y rellena NaN -> 'missing'
    - Si una columna tiene demasiadas categorías, se quedan las top-(max-1) y el resto -> 'OTHER'
    - Además, toda categoría con conteo < rare_min_count -> 'OTHER'
    """
    cat = cat.fillna("missing").astype(str).copy()

    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False)

        # 1) Recortar a top categorías si hay demasiadas
        if len(vc) > max_modalities_per_col:
            keep = set(vc.index[:max_modalities_per_col - 1])  # deja espacio para OTHER
            cat[c] = np.where(cat[c].isin(keep), cat[c], "OTHER")
            vc = cat[c].value_counts(dropna=False)  # recomputa tras recorte

        # 2) Enviar categorías muy raras a OTHER
        rare_vals = set(vc[vc < rare_min_count].index)
        if rare_vals:
            cat[c] = cat[c].where(~cat[c].isin(rare_vals), "OTHER")

    return cat

def safe_explained_inertia(mca) -> np.ndarray:
    """Compatibilidad multi-versión de prince para obtener inercia explicada."""
    if hasattr(mca, "explained_inertia_"):
        return np.array(mca.explained_inertia_, dtype=float)
    if hasattr(mca, "eigenvalues_"):
        ev = np.array(mca.eigenvalues_, dtype=float).ravel()
        total = ev.sum() if ev.sum() != 0 else 1.0
        return ev / total
    if hasattr(mca, "singular_values_"):
        sv = np.array(mca.singular_values_, dtype=float).ravel()
        ev = sv ** 2
        total = ev.sum() if ev.sum() != 0 else 1.0
        return ev / total
    # si no hay nada de lo anterior:
    raise AttributeError("No fue posible obtener la inercia explicada de 'prince.MCA'.")

def fit_mca_full(cat_df: pd.DataFrame, threshold: float = 0.80, n_components: int = 50,
                 max_modalities_per_col: int = 50, rare_min_count: int = 30):
    """
    Ajusta MCA en TODO el dataset (sin muestreo) pero con reducción de cardinalidad para ser más rápido/estable.
    """
    if cat_df.shape[1] == 0:
        return pd.DataFrame(index=cat_df.index), None, np.array([])

    if not PRINCE_OK:
        raise ImportError(
            "No se pudo importar 'prince' (requerido para MCA). "
            "Instala con: pip install prince\nDetalle: " + PRINCE_ERR
        )

    cat = reduce_categorical_cardinality(cat_df, max_modalities_per_col=max_modalities_per_col,
                                         rare_min_count=rare_min_count)

    mca = MCA(n_components=n_components)
    mca = mca.fit(cat)

    inertia = safe_explained_inertia(mca)
    cum_inertia = np.cumsum(inertia)
    ndims = int(np.searchsorted(cum_inertia, threshold) + 1)
    ndims = min(ndims, len(inertia))

    coords = mca.transform(cat)
    ndims = min(ndims, coords.shape[1])
    dims = coords.iloc[:, :ndims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(ndims)]
    return dims, mca, cum_inertia

# ================= Interfaz =================

st.set_page_config(page_title="PCA + MCA | Diabetes", layout="wide")

st.title("PCA + MCA del dataset de Diabetes en EE. UU.")
st.markdown(
    " **Objetivo:** aplicar **PCA** a variables **numéricas** y **MCA** a **categóricas**, "
    "retener componentes/dimensiones hasta alcanzar **≥ 80%** de varianza/inercia acumulada, "
    "y **crear un nuevo dataset** con las **PCs** y las **Dimensiones** concatenadas."
)

with st.expander("🧩 Requisitos e instrucciones rápidas", expanded=True):
    st.markdown("- Dependencias: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `prince`")
    st.code("pip install -r requirements.txt", language="bash")
    st.markdown("- Ejecuta la app con:")
    st.code("streamlit run app.py", language="bash")

DATA_PATH = "diabetic_data.csv"
if not os.path.exists(DATA_PATH):
    st.error("No se encontró `diabetic_data.csv` en el mismo directorio que `app.py`.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.exception(e)
    st.stop()

st.success(f"Dataset cargado: {DATA_PATH} — Forma: {df.shape[0]} filas × {df.shape[1]} columnas")
st.dataframe(df.head(10), use_container_width=True)

# Paso 1: Limpieza IDs
st.header("Paso 1: Limpieza de posibles columnas identificadoras")
try:
    df_clean, dropped = drop_identifier_like_columns(df, high_uniqueness_ratio=0.95)
    st.write("Columnas eliminadas (heurística):", dropped if dropped else "Ninguna")
    st.code("df_clean, dropped = drop_identifier_like_columns(df, high_uniqueness_ratio=0.95)", language="python")
    st.dataframe(df_clean.head(5), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 2: Separación
st.header("Paso 2: Separación de variables numéricas y categóricas")
try:
    num_df, cat_df = split_numeric_categorical(df_clean)
    st.write(f"Numéricas: {num_df.shape[1]} columnas | Categóricas: {cat_df.shape[1]} columnas")
    st.code("num_df, cat_df = split_numeric_categorical(df_clean)", language="python")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vista numéricas")
        st.dataframe(num_df.head(5), use_container_width=True)
    with c2:
        st.subheader("Vista categóricas")
        st.dataframe(cat_df.head(5), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 3: PCA
st.header("Paso 3: PCA sobre variables numéricas (umbral 80%)")
try:
    pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)
    if pca_model is None or pcs.shape[1] == 0:
        st.warning("No se generaron PCs (¿no hay columnas numéricas?).")
    else:
        st.write(f"Componentes retenidas: **{pcs.shape[1]}**")
        st.code("pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)", language="python")
        st.line_chart(pd.DataFrame(cum_var, columns=["Varianza acumulada"]))
        st.dataframe(pcs.head(10), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 4: MCA
st.header("Paso 4: MCA sobre variables categóricas (umbral 80%)")
try:
    if not PRINCE_OK:
        raise ImportError("`prince` no está instalado correctamente. Ejecuta: pip install -r requirements.txt")

    with st.spinner("Calculando MCA (reduciendo categorías para acelerar)..."):
        dims, mca_model, cum_inertia = fit_mca_full(
            cat_df,
            threshold=0.80,
            n_components=50,          # suficientes dims para llegar al 80%
            max_modalities_per_col=50,  # recorta cardinalidad por columna
            rare_min_count=30
        )

    if mca_model is None or dims.shape[1] == 0:
        st.warning("No se generaron Dimensiones (¿no hay columnas categóricas?).")
    else:
        st.write(f"Dimensiones retenidas: **{dims.shape[1]}**")
        st.code(
            "dims, mca_model, cum_inertia = fit_mca_full(cat_df, threshold=0.80, n_components=50, max_modalities_per_col=50, rare_min_count=30)",
            language="python"
        )
        st.line_chart(pd.DataFrame(cum_inertia, columns=["Inercia acumulada"]))
        st.dataframe(dims.head(10), use_container_width=True)

except Exception as e:
    st.exception(e)
    # Si falla MCA, seguimos con PCs solamente
    dims = pd.DataFrame(index=df.index)

# Paso 5: Concatenación
st.header("Paso 5: Nuevo dataset con PCs + Dimensiones concatenadas")
try:
    out_df = pd.concat([pcs, dims], axis=1)
    if out_df.shape[1] == 0:
        st.error("No hay columnas transformadas para mostrar.")
    else:
        st.success(f"Dataset final: {out_df.shape[0]} filas × {out_df.shape[1]} columnas")
        st.dataframe(out_df.head(50), use_container_width=True)

        buf = io.BytesIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Descargar dataset concatenado (CSV)",
            data=buf,
            file_name="dataset_pca_mca.csv",
            mime="text/csv",
        )
except Exception as e:
    st.exception(e)

st.markdown("---")
st.caption("Umbral fijo al 80%. Para datasets muy grandes, la reducción de categorías acelera el MCA sin perder interpretabilidad.")

