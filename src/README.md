# NHANES‑Streamlit

**Análisis interactivo de datos NHANES con Streamlit**  
Aplicación web para explorar prevalencia de condiciones crónicas utilizando **PCA** y **ACM**, con técnicas avanzadas de selección de variables y validación cruzada.

---

##  Contenido del repositorio

- `app_pca_acm_nhanes.py`  
  Aplicación principal en Streamlit. Incluye filtros interactivos, pipelines de procesamiento, análisis de componentes principales (PCA), análisis de correspondencias múltiples (ACM), distintos métodos de selección de variables y validación cruzada.

- `requirements.txt`  
  Paquetes necesarios para ejecutar la app (Streamlit, scikit-learn, imbalanced-learn, prince, seaborn, matplotlib, numpy, pandas, etc.).

- `README.md`  
  Documentación completa del proyecto.

---

##  Descripción de los análisis

### 1. **Preprocesamiento y Pipelines**
- Imputación de valores faltantes con `SimpleImputer`.
- Escalado de variables numéricas con `StandardScaler`.
- Codificación de variables categóricas a través de `OneHotEncoder`.
- Pipeline estructurado con `imblearn` para balanceo de clases mediante SMOTE.

### 2. **Selección de variables**
Implementa tres metodologías:
- **Filtro (SelectKBest)**: Selecciona variables numéricas con mayor relación estadística (ANOVA) con la clase objetivo.
- **Embebido (Random Forest)**: Utiliza la importancia de atributos generada por un bosque aleatorio.
- **Envoltura (RFE)**: Selecciona variables óptimas retroactivamente usando un modelo (p. ej. Regresión Logística).

### 3. **Validación cruzada**
Para cada método de selección se evalúa la capacidad predictiva mediante `cross_val_score` sobre un clasificador (Regresión Logística). Así se mide la consistencia y evita overfitting.

### 4. **PCA (Análisis de Componentes Principales)**
- Reducción de dimensionalidad tras aplicar SMOTE y selección RFE.
- Visualización gráfica del espacio PCA con colores por condición.
- Cálculo y visualización de **loadings**, mostrando la contribución relativa de cada variable original a cada componente.

### 5. **ACM (Análisis de Correspondencias Múltiples)**
- Aplicado a variables categóricas (como sexo, raza, edad).
- Visualiza asociaciones entre categorías mediante `prince.MCA` o `mca`.

---

##  Cómo ejecutar la aplicación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/jflorez-giraldo/Nhanes-streamlit.git
   cd Nhanes-streamlit
