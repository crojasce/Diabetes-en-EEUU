
# (Opcional) crear y activar un entorno virtual
# python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
# venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows

# conda
conda activate mi_entorno

pip install prince

pip install -r requirements.txt
streamlit run app.py
