import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DATA_PATH = "../diabetic_data.csv"   # Ruta al archivo CSV
OUTPUT_PATH = "../dataset_pca_mca.csv"  # Salida con resultados
VAR_THRESHOLD = 0.90  # 90% de varianza acumulada
ID_COLS = ["encounter_id", "patient_nbr"]

# -----------------------------
# 1. Cargar datos
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Separar columnas numéricas y categóricas
# -----------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Quitar columnas ID
num_cols = [c for c in num_cols if c not in ID_COLS]
cat_cols = [c for c in cat_cols if c not in ID_COLS]

# -----------------------------
# 3. PCA para variables numéricas
# -----------------------------
if num_cols:
    imputer_num = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num = imputer_num.fit_transform(df[num_cols])
    X_num_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=VAR_THRESHOLD, svd_solver="full", random_state=42)
    Z_num = pca.fit_transform(X_num_scaled)
    pca_cols = [f"PCA_{i+1}" for i in range(Z_num.shape[1])]
    pca_df = pd.DataFrame(Z_num, columns=pca_cols)
else:
    pca_df = pd.DataFrame()

# -----------------------------
# 4. MCA para variables categóricas (OneHot + TruncatedSVD)
# -----------------------------
if cat_cols:
    X_cat_raw = df[cat_cols].replace("?", np.nan)
    X_cat_filled = X_cat_raw.fillna("Unknown")

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    X_cat_ohe = ohe.fit_transform(X_cat_filled)

    svd = TruncatedSVD(n_components=min(200, X_cat_ohe.shape[1]-1), random_state=42)
    svd.fit(X_cat_ohe)
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    k_cat = np.searchsorted(cum_var, VAR_THRESHOLD) + 1

    svd_final = TruncatedSVD(n_components=k_cat, random_state=42)
    Z_cat = svd_final.fit_transform(X_cat_ohe)
    mca_cols = [f"MCA_{i+1}" for i in range(k_cat)]
    mca_df = pd.DataFrame(Z_cat, columns=mca_cols)
else:
    mca_df = pd.DataFrame()

# -----------------------------
# 5. Unir resultados y guardar
# -----------------------------
final_df = pd.concat([pca_df, mca_df], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Componentes PCA: {pca_df.shape[1]}")
print(f"Dimensiones MCA: {mca_df.shape[1]}")
print(f"Archivo guardado en: {OUTPUT_PATH}")
