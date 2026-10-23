# Ejecutar el workshop en local (VS Code / Antigravity)

Los notebooks fueron escritos para Google Colab, pero corren igual en local.
Ya traen un *fallback*: si no detectan Colab, leen la `NVIDIA_API_KEY` desde la
variable de entorno o un archivo `.env`, y si no la encuentran la piden por
consola (`getpass`). No hay que cambiar código para la key.

## 1. Requisitos

- **Python 3.11 o 3.12** (recomendado). Comprueba con `python --version`.
- VS Code con la extensión **Python** + **Jupyter** (o Antigravity con soporte de
  notebooks).

## 2. Crear entorno virtual e instalar dependencias

Desde la carpeta `Semana10-WorkshopRAG/`:

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> La primera instalación tarda un poco (gradio, faiss, markitdown). Si
> `faiss-cpu` falla en tu sistema, puedes omitirlo: solo lo usa el reto C1.

## 3. Configurar la API key (sin Colab)

1. Copia `.env.example` a `.env`.
2. Pega tu clave real (la generas gratis en https://build.nvidia.com/):
   ```
   NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Los notebooks ya hacen `load_dotenv()`, así que tomarán la clave del `.env`
   automáticamente. **No** escribas la clave dentro de ninguna celda.

El archivo `.env` está en `.gitignore`: no se sube al repo.

## 4. Seleccionar el kernel

Al abrir un notebook en VS Code / Antigravity, elige el kernel del entorno
`.venv` que creaste (arriba a la derecha → "Select Kernel" → Python `.venv`).

## 5. Rutas de archivos (local vs Colab)

- **Notebook 02b (A2):** el código ejecuta `load_documents(Path("company_documents.json"))`.
  En local, la forma más simple es **abrir el notebook desde la carpeta
  `data/`**, o cambiar esa línea por
  `load_documents(Path("data/company_documents.json"))`.
  Dejé una nota en esa celda. Si el archivo no se encuentra, el notebook usa un
  corpus de demostración y lo avisa por consola.
- **Notebook 03 (papers):** descarga los PDFs de arXiv a `papers/` la primera
  vez (necesitas internet). Luego los reutiliza.
- **Notebook 04 (C4):** busca `preguntas.csv` y, si no, `eval/preguntas.csv`.
  Ábrelo desde la raíz del workshop para que encuentre `eval/preguntas.csv`.
- **Notebook 04 (C6):** crea `qdrant_data/` en disco (persistencia). Está en
  `.gitignore`.

## 6. Orden de ejecución

1. `notebooks/02b_rag_app_nvidia_nim_colab.ipynb` → tareas A1–A4. La última celda
   abre Gradio en `http://127.0.0.1:7860` (en local no necesitas `share=True`,
   pero funciona igual).
2. `notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb` → B1–B5 y R1.
   **Ejecuta las Partes 0–4 antes** de correr B2–B5.
3. `notebooks/04_challenges.ipynb` → retos. Varias celdas reutilizan variables
   del notebook 03 (`embeddings`, `all_docs`, `vectorstore`, `build_store`...).
   Para C4/C5/C6 ejecuta el notebook 03 primero en la misma sesión, o copia las
   celdas de carga necesarias.

## 7. Antes de entregar

- Rellena los `[PENDIENTE]` de `RESPUESTAS.md` con las salidas reales.
- Guarda las capturas pedidas en `evidencias/`.
- Busca `nvapi` en los `.ipynb` (incluyendo salidas guardadas) y borra cualquier
  rastro de la clave. En VS Code: "Clear All Outputs" antes de guardar/commitear.
- Completa tu código estudiantil en `README.md` y `RESPUESTAS.md`.
