# app.py
# ------------------------------------------------------------
# PCA (numéricas) + MCA (categóricas) con umbral del 80%
# Muestra paso a paso y entrega dataset final concatenado.
#
# Requisitos:
#   pip install streamlit pandas numpy scikit-learn prince
# Ejecuta:
#   streamlit run app.py
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

# =============== Utilidades ===============

def is_identifier(colname: str) -> bool:
    """Heurística simple para detectar columnas tipo ID por nombre."""
    name = str(colname).lower()
    return any(tok in name for tok in ["id", "identifier", "uuid", "nbr", "key", "encounter"])

def drop_identifier_like_columns(
    df: pd.DataFrame,
    high_uniqueness_ratio: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Elimina columnas que parecen identificadores:
    - Por nombre (contiene 'id', 'uuid', etc.)
    - Por unicidad (nunique muy cercano a número de filas)
    """
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
    """Separa variables numéricas y categóricas según dtype."""
    num_df = df.select_dtypes(include=["number"]).copy()
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    return num_df, cat_df

def fit_pca(num_df: pd.DataFrame, threshold: float = 0.80):
    """Estandariza, ajusta PCA y devuelve PCs hasta alcanzar el umbral."""
    if num_df.shape[1] == 0:
        return pd.DataFrame(index=num_df.index), None, np.array([])

    X = num_df.fillna(num_df.mean(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    Xp = pca.fit_transform(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cum_var, threshold) + 1)

    pcs = pd.DataFrame(Xp[:, :ncomp], index=num_df.index,
                       columns=[f"PC{i+1}" for i in range(ncomp)])
    return pcs, pca, cum_var

def fit_mca(cat_df: pd.DataFrame, threshold: float = 0.80, sample_rows: int = 20000, rare_min_count: int = 10):
    """
    Ajusta MCA con 'prince' de forma eficiente:
      - Agrupa categorías raras en 'OTHER' (por columna).
      - Ajusta el modelo en una muestra (hasta sample_rows).
      - Aplica el modelo a TODO el dataset para obtener coordenadas.
    Devuelve:
      dims (DataFrame), mca (modelo), cum_inertia (np.array)
    """
    if cat_df.shape[1] == 0:
        return pd.DataFrame(index=cat_df.index), None, np.array([])

    if not PRINCE_OK:
        raise ImportError(
            "No se pudo importar 'prince' (requerido para MCA). "
            "Instala con: pip install prince\nDetalle: " + PRINCE_ERR
        )

    # Prepara categóricas: str + missing
    cat = cat_df.fillna("missing").astype(str).copy()

    # Agrupar categorías raras por columna para reducir cardinalidad
    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False)
        rare_vals = set(vc[vc < rare_min_count].index)
        if rare_vals:
            cat[c] = cat[c].where(~cat[c].isin(rare_vals), "OTHER")

    # Muestra para entrenar MCA (si hay muchas filas)
    if sample_rows and len(cat) > sample_rows:
        cat_fit = cat.sample(sample_rows, random_state=42)
    else:
        cat_fit = cat

    # Ajuste en la muestra
    mca = MCA()
    mca = mca.fit(cat_fit)

    # Inercia acumulada y selección de dimensiones
    inertia = np.array(mca.explained_inertia_)
    cum_inertia = np.cumsum(inertia)
    ndims = int(np.searchsorted(cum_inertia, threshold) + 1)

    # Transformar TODO el dataset con el modelo entrenado
    coords_all = mca.transform(cat)
    dims = coords_all.iloc[:, :ndims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(ndims)]
    return dims, mca, cum_inertia

# =============== Interfaz ===============

st.set_page_config(page_title="PCA + MCA | Diabetes", layout="wide")

st.title("PCA + MCA del dataset de Diabetes en EE. UU.")
st.markdown(
    " **Objetivo:** aplicar **PCA** a variables **numéricas** y **MCA** a **categóricas**, "
    "retener componentes/dimensiones hasta alcanzar **≥ 80%** de varianza/inercia acumulada, "
    "y **crear un nuevo dataset** con las **PCs** y las **Dimensiones** concatenadas."
)

with st.expander("🧩 Requisitos e instrucciones rápidas", expanded=True):
    st.markdown("- Dependencias: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `prince`")
    st.code("pip install streamlit pandas numpy scikit-learn prince", language="bash")
    st.markdown("- Ejecuta la app con:")
    st.code("streamlit run app.py", language="bash")

# Carga simple: el archivo debe estar junto a app.py
DATA_PATH = "diabetic_data.csv"
if not os.path.exists(DATA_PATH):
    st.error(
        "No se encontró `diabetic_data.csv` en el mismo directorio que `app.py`.\n"
        "Cópialo ahí y recarga la página."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success(f"Dataset cargado: {DATA_PATH} — Forma: {df.shape[0]} filas × {df.shape[1]} columnas")
st.dataframe(df.head(10), use_container_width=True)

# Paso 1: remover posibles IDs
st.header("Paso 1: Limpieza de posibles columnas identificadoras")
df_clean, dropped = drop_identifier_like_columns(df, high_uniqueness_ratio=0.95)
st.write("Columnas eliminadas (heurística):", dropped if dropped else "Ninguna")
st.code(
    "df_clean, dropped = drop_identifier_like_columns(df, high_uniqueness_ratio=0.95)",
    language="python"
)
st.dataframe(df_clean.head(5), use_container_width=True)

# Paso 2: separar por tipo
st.header("Paso 2: Separación de variables numéricas y categóricas")
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

# Paso 3: PCA numéricas (umbral 80%)
st.header("Paso 3: PCA sobre variables numéricas (umbral 80%)")
pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)
if pca_model is None or pcs.shape[1] == 0:
    st.warning("No se generaron PCs (¿no hay columnas numéricas?).")
else:
    st.write(f"Componentes retenidas: **{pcs.shape[1]}**")
    st.code(
        "pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)",
        language="python"
    )
    st.line_chart(pd.DataFrame(cum_var, columns=["Varianza acumulada"]))
    st.dataframe(pcs.head(10), use_container_width=True)

# Paso 4: MCA categóricas (umbral 80%)
st.header("Paso 4: MCA sobre variables categóricas (umbral 80%)")
try:
    with st.spinner("Calculando MCA (esto puede tardar unos segundos)..."):
        # Puedes ajustar sample_rows y rare_min_count si quieres más/menos precisión/velocidad
        dims, mca_model, cum_inertia = fit_mca(cat_df, threshold=0.80, sample_rows=20000, rare_min_count=10)

    if mca_model is None or dims.shape[1] == 0:
        st.warning("No se generaron Dimensiones (¿no hay columnas categóricas?).")
    else:
        st.write(f"Dimensiones retenidas: **{dims.shape[1]}**")
        st.code("dims, mca_model, cum_inertia = fit_mca(cat_df, threshold=0.80, sample_rows=20000, rare_min_count=10)", language="python")
        st.line_chart(pd.DataFrame(cum_inertia, columns=["Inercia acumulada"]))
        st.dataframe(dims.head(10), use_container_width=True)
except ImportError as e:
    st.warning(str(e))
    dims = pd.DataFrame(index=df.index)


# Paso 5: Concatenación final
st.header("Paso 5: Nuevo dataset con PCs + Dimensiones concatenadas")
out_df = pd.concat([pcs, dims], axis=1)
if out_df.shape[1] == 0:
    st.error("No hay columnas transformadas para mostrar. Revisa que existan numéricas/categóricas.")
else:
    st.success(f"Dataset final: {out_df.shape[0]} filas × {out_df.shape[1]} columnas")
    st.dataframe(out_df.head(50), use_container_width=True)

    # Descarga CSV
    buf = io.BytesIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        label="⬇️ Descargar dataset concatenado (CSV)",
        data=buf,
        file_name="dataset_pca_mca.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Umbral fijo al 80% para simplicidad. Ajusta el código si deseas hacerlo configurable.")




