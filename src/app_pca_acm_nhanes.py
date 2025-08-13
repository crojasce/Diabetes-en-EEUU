import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from mca import MCA  # Asegúrate de tener instalada la librería: pip install mca
from sklearn.base import BaseEstimator, TransformerMixin
import prince
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="PCA Streamlit App", layout="wide")
st.title("PCA and MCA Analysis with NHANES Data")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/nhanes_2015_2016.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Asignar condiciones
def assign_condition(row):
    if (row["BPXSY1"] >= 140 or row["BPXSY2"] >= 140 or 
        row["BPXDI1"] >= 90 or row["BPXDI2"] >= 90):
        return "hypertension"
    elif row["BMXBMI"] >= 30:
        return "diabetes"
    elif ((row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or 
          (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88)):
        return "high cholesterol"
    else:
        return "healthy"

df["Condition"] = df.apply(assign_condition, axis=1)

# Diccionario de códigos por variable categórica
category_mappings = {
    "RIAGENDR": {
        1: "Male",
        2: "Female"
    },
    "DMDMARTL": {
        1: "Married",
        2: "Divorced",
        3: "Never married",
        4: "Widowed",
        5: "Separated",
        6: "Living with partner",
        77: "Refused",
        99: "Don't know"
    },
    "DMDEDUC2": {
        1: "Less than 9th grade",
        2: "9-11th grade (no diploma)",
        3: "High school/GED",
        4: "Some college or AA degree",
        5: "College graduate or above",
        7: "Refused",
        9: "Don't know"
    },
    "SMQ020": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ101": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ110": {
        1: "Every day",
        2: "5–6 days/week",
        3: "3–4 days/week",
        4: "1–2 days/week",
        5: "2–3 days/month",
        6: "Once a month or less",
        7: "Refused",
        9: "Don't know"
    },
    "RIDRETH1": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race - Including Multi-Racial"
    },
    "DMDCITZN": {
        1: "Citizen by birth or naturalization",
        2: "Not a citizen of the U.S.",
        7: "Refused",
        9: "Don't know"
    },
    "HIQ210": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "SDMVPSU": {
        1: "PSU 1",
        2: "PSU 2"
    },
    "DMDHHSIZ": {
        1: "1 person",
        2: "2 people",
        3: "3 people",
        4: "4 people",
        5: "5 people",
        6: "6 people",
        7: "7 or more people"
    }
}

def apply_categorical_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

df = apply_categorical_mappings(df, category_mappings)

col_map = {
    "SEQN": "Participant ID",
    "ALQ101": "Alcohol Intake - Past 12 months (Q1)",
    "ALQ110": "Alcohol Frequency",
    "ALQ130": "Alcohol Amount",
    "SMQ020": "Smoking Status",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age (years)",
    "RIDRETH1": "Race/Ethnicity",
    "DMDCITZN": "Citizenship",
    "DMDEDUC2": "Education Level",
    "DMDMARTL": "Marital Status",
    "DMDHHSIZ": "Household Size",
    "WTINT2YR": "Interview Weight",
    "SDMVPSU": "Masked PSU",
    "SDMVSTRA": "Masked Stratum",
    "INDFMPIR": "Income to Poverty Ratio",
    "BPXSY1": "Systolic BP1",
    "BPXDI1": "Diastolic BP1",
    "BPXSY2": "Systolic BP2",
    "BPXDI2": "Diastolic BP2",
    "BMXWT": "Body Weight",
    "BMXHT": "Body Height",
    "BMXBMI": "Body Mass Index",
    "BMXLEG": "Leg Length",
    "BMXARML": "Arm Length",
    "BMXARMC": "Arm Circumference",
    "BMXWAIST": "Waist Circumference",
    "HIQ210": "Health Insurance Coverage"
}

df = df.rename(columns=col_map)

# Asegurar compatibilidad con Arrow/Streamlit
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("string").fillna("Missing")

df = df.reset_index(drop=True)

st.markdown("""
## Sobre el Dataset NHANES 2015-2016

El **National Health and Nutrition Examination Survey (NHANES)** es un estudio realizado por el Centro Nacional de Estadísticas de Salud (NCHS) de Estados Unidos. Su objetivo principal es evaluar la salud y el estado nutricional de adultos y niños en la población estadounidense mediante entrevistas y exámenes físicos.

### Detalles del Dataset usado:

- Este conjunto de datos corresponde al ciclo **2015-2016** de NHANES.
- Contiene variables demográficas, clínicas, de comportamiento, biométricas y de salud, tales como:
  - Edad, sexo y raza/etnia.
  - Medidas antropométricas: peso, altura, índice de masa corporal (IMC), circunferencia de cintura.
  - Mediciones de presión arterial.
  - Información sobre consumo de alcohol y tabaco.
  - Datos sobre educación, estado civil y nivel socioeconómico.
- Este dataset es utilizado para analizar condiciones de salud como hipertensión, diabetes y niveles altos de colesterol, entre otros.

### Propósito del Análisis en esta App:

- Clasificar a los participantes en grupos de salud: **hipertensión**, **diabetes**, **alto colesterol** y **saludable**, basados en indicadores clínicos.
- Realizar un análisis exploratorio con técnicas de reducción de dimensionalidad como:
  - PCA (Análisis de Componentes Principales) para variables numéricas.
  - MCA (Análisis de Correspondencias Múltiples) para variables categóricas.
- Seleccionar las variables más relevantes para la clasificación usando técnicas estadísticas y de machine learning.

### Importancia:

NHANES es un recurso valioso para investigadores, médicos y políticas públicas que buscan entender factores de riesgo y prevalencia de enfermedades crónicas en la población estadounidense. Este análisis ayuda a identificar patrones clave en los datos que pueden guiar intervenciones de salud.

---

**Fuente:** [NHANES 2015-2016](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015)
""")


# Mostrar info y variables categóricas lado a lado
st.subheader("Resumen de Datos")

# Crear columnas para mostrar info_df y category_df lado a lado
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tipo de Dato y Nulos**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Dtype": df.dtypes.values
    })
    st.dataframe(info_df, use_container_width=True)

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].dtype == 'string' or 
                       df[col].nunique() <= 10]

with col2:
    st.markdown("**Variables Categóricas Detectadas**")
    category_info = []
    for col in categorical_vars:
        unique_vals = df[col].dropna().unique()
        category_info.append({
            "Variable": col,
            "Unique Classes": ", ".join(map(str, sorted(unique_vals)))
        })

    category_df = pd.DataFrame(category_info)
    st.dataframe(category_df, use_container_width=True)


st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# Filtros
with st.sidebar:
    st.header("Filters")
    gender_filter = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()))
    race_filter = st.multiselect("Race/Ethnicity", sorted(df["Race/Ethnicity"].dropna().unique()))
    condition_filter = st.multiselect("Condition", sorted(df["Condition"].dropna().unique()))
    #st.markdown("---")
    #k_vars = st.slider("Number of variables to select", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Gender": gender_filter, "Race/Ethnicity": race_filter, "Condition": condition_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# Mostrar advertencias
problematic_cols = df.columns[df.dtypes == "object"].tolist()
nullable_ints = df.columns[df.dtypes.astype(str).str.contains("Int64")].tolist()

st.write("### ⚠️ Columnas potencialmente problemáticas para Arrow/Streamlit:")
if problematic_cols or nullable_ints:
    st.write("**Tipo 'object':**", problematic_cols)
    st.write("**Tipo 'Int64' (nullable):**", nullable_ints)
else:
    st.success("✅ No hay columnas problemáticas detectadas.")

# Separar variables
#target_col = "Condition"

# Evitar que la variable objetivo quede en X
#X = df.drop(columns=[target_col])
y = df["Condition"]


numeric_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if 'Condition' in numeric_features:
    numeric_features.remove('Condition')

X_num = df[numeric_features]
X_cat = df[categorical_vars]

st.markdown("""
## ¿Qué es el Análisis de Componentes Principales (PCA)?

El **Análisis de Componentes Principales (PCA)** es una técnica estadística usada para reducir la dimensionalidad de grandes conjuntos de datos con muchas variables numéricas, manteniendo la mayor cantidad posible de la variabilidad original.

### ¿Cómo funciona?

- PCA transforma las variables originales en un nuevo conjunto de variables no correlacionadas llamadas **componentes principales**.
- Cada componente principal es una combinación lineal de las variables originales.
- Los primeros componentes capturan la mayor parte de la variabilidad en los datos.
- Esto permite visualizar y analizar datos complejos en menos dimensiones, facilitando la interpretación.

### ¿Por qué usamos PCA en este análisis?

- Para explorar patrones y agrupamientos en las variables numéricas del dataset NHANES.
- Para visualizar relaciones entre individuos y variables de forma simplificada.
- Para identificar cuáles variables contribuyen más a la variabilidad total.

---
""")

# Pipeline con imputación, escalado y PCA
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10))
])
X_num_pca = numeric_pipeline.fit_transform(X_num)

# DataFrame con componentes
pca_df = pd.DataFrame(X_num_pca, columns=[f"PC{i+1}" for i in range(10)])
pca_df["Condition"] = y.values

# Extraer el objeto PCA del pipeline
pca = numeric_pipeline.named_steps["pca"]

# Obtener la varianza explicada de cada componente
explained_variance_ratio = pca.explained_variance_ratio_

# Calcular la varianza acumulada
cumulative_variance = np.cumsum(explained_variance_ratio)

# Graficar
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
ax.set_xlabel("Número de Componentes Principales")
ax.set_ylabel("Varianza Acumulada")
ax.set_title("Varianza Acumulada de las Componentes Principales")
ax.set_xticks(range(1, len(cumulative_variance) + 1))
ax.grid(True)
st.pyplot(fig)

# Gráfico PCA PC1 vs PC2
st.subheader("PCA - PC1 vs PC2")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", palette="Set2", alpha=0.8, ax=ax)
ax.set_title("PCA - PC1 vs PC2")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

st.markdown("""
### ¿Qué son los Loadings en PCA?

En el Análisis de Componentes Principales (PCA), los **loadings** (o cargas factoriales) representan la relación entre las variables originales y las componentes principales.

---

### Cómo se calculan

PCA busca nuevas variables llamadas **componentes principales**, que son combinaciones lineales de las variables originales.

Cada componente principal $PC_j$ se define como:

$$
PC_j = a_{1j} x_1 + a_{2j} x_2 + \cdots + a_{pj} x_p
$$

donde:

- $x_1, x_2, \ldots, x_p$ son las variables originales (normalizadas si se aplica escalado),
- $a_{ij}$ son los coeficientes o **loadings** de la variable $i$ en la componente $j$.

---

### Interpretación

- El loading $a_{ij}$ indica **cuánto aporta** la variable $i$ a la componente principal $j$.
- Valores altos (en valor absoluto) significan que esa variable influye fuertemente en la componente.
- El signo indica la dirección de la relación (positiva o negativa).

---

### Cálculo práctico

- Los loadings corresponden a los **autovectores** (vectores propios) de la matriz de covarianza o correlación de los datos.
- En `scikit-learn`, el atributo `components_` del objeto PCA contiene estos loadings:  
  - Cada fila es una componente principal,  
  - Cada columna es una variable original.

---

### Usos

- Analizar los loadings ayuda a interpretar qué variables definen cada componente.
- También es útil para seleccionar variables importantes según su contribución.
""")

# Obtener los loadings del PCA (componentes * características)
loadings = numeric_pipeline.named_steps["pca"].components_

# Convertir a DataFrame con nombres de columnas
loadings_df = pd.DataFrame(
    loadings,
    columns=X_num.columns,
    index=[f"PC{i+1}" for i in range(loadings.shape[0])]
).T  # Transponer para que columnas sean PCs y filas las variables

# Ordenar las filas por la importancia de la variable en la suma de cuadrados de los componentes
# Esto agrupa por aquellas variables con mayor contribución total
loading_magnitude = (loadings_df**2).sum(axis=1)
loadings_df["Importance"] = loading_magnitude
loadings_df_sorted = loadings_df.sort_values(by="Importance", ascending=False).drop(columns="Importance")

# Graficar heatmap ordenado
st.subheader("🔍 Heatmap de Loadings del PCA (Componentes Principales)")

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(loadings_df_sorted, annot=True, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

st.markdown("""
### ¿Qué es el Análisis de Correspondencias Múltiples (MCA)?

El Análisis de Correspondencias Múltiples (MCA) es una técnica estadística exploratoria utilizada para analizar y visualizar datos categóricos. Es una extensión del Análisis de Correspondencias Simple (CA) cuando hay más de dos variables categóricas.

**Objetivos principales del MCA:**

- **Reducir la dimensionalidad** de datos categóricos complejos.
- **Identificar patrones** y relaciones entre categorías de variables.
- **Visualizar** asociaciones entre individuos y categorías en un espacio de menor dimensión.

**¿Cómo funciona?**

El MCA transforma las variables categóricas en un espacio numérico, similar al Análisis de Componentes Principales (PCA) para variables numéricas. Luego, representa las observaciones y categorías en un mapa factorial bidimensional o tridimensional, donde la proximidad entre puntos indica similitudes.

**Aplicaciones comunes:**

- Encuestas y estudios sociales con muchas variables categóricas.
- Análisis de perfiles de consumidores.
- Estudios epidemiológicos para agrupar características clínicas o sociodemográficas.

En resumen, MCA es una herramienta poderosa para explorar y resumir grandes conjuntos de datos categóricos y facilita la interpretación visual de relaciones complejas.
""")

# Pipeline para MCA con prince
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

# Transformar datos categóricos
X_cat_encoded = categorical_pipeline.fit_transform(X_cat)

# Crear DataFrame con nombres de columnas después del one-hot encoding
encoded_columns = categorical_pipeline.named_steps["encoder"].get_feature_names_out(X_cat.columns)
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoded_columns, index=X_cat.index)

# Aplicar MCA con prince
mca = prince.MCA(n_components=2, random_state=42)
X_cat_mca = mca.fit_transform(X_cat_encoded_df)

# Agregar columna de condición para colorear
mca_df = X_cat_mca.copy()
mca_df["Condition"] = y.values
mca_df.columns = ["Dim1", "Dim2", "Condition"]

st.subheader("MCA - Dim1 vs Dim2")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=mca_df, x="Dim1", y="Dim2", hue="Condition", palette="Set1", alpha=0.8, ax=ax)
ax.set_title("MCA - Dim1 vs Dim2")
ax.set_xlabel("Dim1")
ax.set_ylabel("Dim2")
st.pyplot(fig)

# ======================
# HEATMAP DE CONTRIBUCIONES EN MCA
# ======================


st.subheader("🔍 Contribuciones de las Variables Categóricas al MCA")

# Obtener contribuciones a las dimensiones
contribs = mca.column_contributions_

#st.write("Column names in contributions DataFrame:", contribs.columns.tolist())

# Seleccionar contribuciones a Dim1 y Dim2
contribs_selected = contribs[[0, 1]]  # 0 = Dim1, 1 = Dim2
contribs_selected.columns = ["Dim1", "Dim2"]

# Ordenar por Dim1 para mejor visualización (opcional)
#contribs_sorted = contribs_selected.sort_values(by=0, ascending=False)
contribs_sorted = contribs_selected.sort_values("Dim1", ascending=False)

# Crear heatmap
fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(contribs_sorted))))
sns.heatmap(contribs_sorted, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
ax.set_title("Contribuciones de las Variables a las Dimensiones del MCA")
st.pyplot(fig)


# ======================
# 🔧 Construir X
# ======================

# Separar por tipo
x_num = df.select_dtypes(include=["float64", "int64"]).drop(columns=["SEQN"], errors="ignore")
x_cat = df.select_dtypes(include=["object", "category", "bool", "string"])

# Guardar nombres originales
num_features = x_num.columns.tolist()
cat_features = x_cat.columns.tolist()

# Preprocesadores
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer combinado
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Ajustar y transformar
preprocessor.fit(df)

# Obtener nombres después del preprocesamiento
cat_encoded_columns = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_features)
feature_names = np.concatenate([num_features, cat_encoded_columns])

# Transformar el dataset
X = preprocessor.transform(df)
X_df = pd.DataFrame(X, columns=feature_names)

# ======================
# 🔍 Selección de Variables
# ======================
st.header("🔍 Selección de Variables")

# Variable objetivo
y = df["Condition"]

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 1️⃣ Selección basada en modelos
# ============================
with st.expander("1️⃣ Selección basada en modelos (Random Forest)"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    st.subheader("Importancia de variables")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances.head(7), y=importances.head(7).index, ax=ax)
    st.pyplot(fig)

    st.write("Precisión en test:", np.round(model.score(X_test, y_test), 3))

# ============================
# 3️⃣ Selección por envoltura
# ============================
with st.expander("3️⃣ Selección por envoltura (RFE con Regresión Logística)"):
    logistic = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(estimator=logistic, n_features_to_select=7)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]

    st.write("Variables seleccionadas:", selected_features.tolist())

    coefs = pd.Series(logistic.fit(X_train[selected_features], y_train).coef_[0], index=selected_features)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=coefs, y=coefs.index, ax=ax)
    st.pyplot(fig)
    
















