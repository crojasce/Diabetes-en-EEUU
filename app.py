# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD

st.set_page_config(page_title="PCA + MCA (Bioestadística)", layout="wide")

st.title("Taller 1 · PCA (num) + MCA (cat)")
st.caption("Realizar PCA (sobre el conjunto de variables numéricas) y MCA (sobre el conjunto de variables categóricas), seleccionar las componentes principales y las dimensiones que acumulen una varianza por encima de algún porcentaje (ustedes lo deciden) y crear un nuevo dataset con las pcs y las dimensiones concatenadas.")
st.caption("Cargue su CSV, elija el umbral de varianza y obtenga el dataset con las componentes concatenadas.")

# -----------------------------
# Sidebar: parámetros
# -----------------------------
with st.sidebar:
    st.header("Parámetros")
    var_threshold = st.slider("Varianza acumulada objetivo", 0.50, 0.99, 0.90, 0.01)
    max_svd_try = st.number_input("Máx. componentes a probar (MCA)", min_value=50, max_value=2000, value=500, step=50)
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)
    st.markdown("---")
    st.caption("IDs que NO se usan en el análisis")
    id_cols_text = st.text_input("Columnas ID (separadas por coma)", "encounter_id,patient_nbr")

# -----------------------------
# Carga de datos
# -----------------------------
st.subheader("1) Cargar datos")
uploaded = st.file_uploader("Sube tu archivo .csv (por ejemplo diabetic_data.csv)", type=["csv"])

if uploaded is None:
    st.info("Esperando un CSV…")
    st.stop()

df = pd.read_csv(uploaded)
st.write("**Vista rápida (primeras filas):**")
st.dataframe(df.head())

# Parsear columnas ID
id_cols = [c.strip() for c in id_cols_text.split(",") if c.strip()]

# -----------------------------
# Detección de tipos
# -----------------------------
st.subheader("2) Detección de variables")
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Mover *_id a categóricas (si aparecen como numéricas)
for c in df.columns:
    if c.endswith("_id") and c not in id_cols and c not in cat_cols:
        if c in num_cols:
            num_cols.remove(c)
        cat_cols.append(c)

# Quitar IDs
num_cols = [c for c in num_cols if c not in id_cols]
cat_cols = [c for c in cat_cols if c not in id_cols]

c1, c2 = st.columns(2)
with c1:
    st.write("**Numéricas:**", len(num_cols))
    st.code(", ".join(num_cols[:25]) + ("..." if len(num_cols) > 25 else ""))
with c2:
    st.write("**Categóricas:**", len(cat_cols))
    st.code(", ".join(cat_cols[:25]) + ("..." if len(cat_cols) > 25 else ""))

# -----------------------------
# PCA numéricas
# -----------------------------
st.subheader("3) PCA en variables numéricas")
if len(num_cols) == 0:
    st.warning("No se detectaron variables numéricas (o quedaron todas excluidas).")
    pca_df = pd.DataFrame(index=df.index)
else:
    Xn = df[num_cols].copy()
    imp = SimpleImputer(strategy="median")
    Xn_imp = imp.fit_transform(Xn)
    scaler = StandardScaler()
    Xn_scaled = scaler.fit_transform(Xn_imp)

    # n_components como proporción para alcanzar el umbral
    pca = PCA(n_components=var_threshold, svd_solver="full", random_state=random_state)
    Z_num = pca.fit_transform(Xn_scaled)
    pca_cols = [f"PCA_{i+1}" for i in range(Z_num.shape[1])]
    pca_df = pd.DataFrame(Z_num, columns=pca_cols, index=df.index)

    st.write(f"**Componentes PCA seleccionadas:** {pca_df.shape[1]}")
    st.line_chart(pd.Series(pca.explained_variance_ratio_).cumsum(), height=180)
    st.dataframe(pca_df.head())

# -----------------------------
# MCA categóricas (OneHot + TruncatedSVD)
# -----------------------------
st.subheader("4) MCA en variables categóricas")
if len(cat_cols) == 0:
    st.warning("No se detectaron variables categóricas.")
    mca_df = pd.DataFrame(index=df.index)
else:
    Xc_raw = df[cat_cols].copy().replace("?", np.nan)
    Xc = Xc_raw.fillna("Unknown")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    Xc_ohe = ohe.fit_transform(Xc)

    # Intento incremental de SVD para alcanzar el umbral (sin reventar memoria)
    n_try = min(max_svd_try, Xc_ohe.shape[1] - 1) if Xc_ohe.shape[1] > 1 else 0
    if n_try <= 0:
        mca_df = pd.DataFrame(index=df.index)
        st.warning("No fue posible construir SVD (muy pocas columnas tras One-Hot).")
    else:
        svd_probe = TruncatedSVD(n_components=n_try, random_state=random_state)
        svd_probe.fit(Xc_ohe)
        cum = svd_probe.explained_variance_ratio_.cumsum()
        k_cat = int(np.searchsorted(cum, var_threshold) + 1)
        k_cat = min(k_cat, n_try)

        svd = TruncatedSVD(n_components=k_cat, random_state=random_state)
        Z_cat = svd.fit_transform(Xc_ohe)
        mca_cols = [f"MCA_{i+1}" for i in range(k_cat)]
        mca_df = pd.DataFrame(Z_cat, columns=mca_cols, index=df.index)

        st.write(f"**Dimensiones MCA seleccionadas:** {mca_df.shape[1]}")
        st.line_chart(pd.Series(svd_probe.explained_variance_ratio_).cumsum(), height=180)
        st.dataframe(mca_df.head())

# -----------------------------
# Concatenación y descarga
# -----------------------------
st.subheader("5) Dataset final (PCA + MCA)")
final_df = pd.concat([pca_df, mca_df], axis=1)
st.write(f"**Forma del dataset final:** {final_df.shape[0]} filas × {final_df.shape[1]} columnas")
st.dataframe(final_df.head())

# Descargar CSV
csv_bytes = final_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar dataset_pca_mca.csv",
    data=csv_bytes,
    file_name="dataset_pca_mca.csv",
    mime="text/csv"
)








