# DIABETES

**ANÁLISIS CON LAS VARIABLES CATEGÓRICAS**  
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



---

##  Cómo ejecutar la aplicación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/croajsce/Diabetes-en-EEUU.git
cd Diabetes-en-EEUU
streamlit run app.py
