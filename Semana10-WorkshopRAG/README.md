# Semana 10 — Workshop RAG (SI3003)

**Autor:** Juan Camilo Gómez · **Usuario GitHub:** `jcgomezj` · **Código:** `[COMPLETAR tu código estudiantil]`

Workshop de *Retrieval-Augmented Generation* con LangChain, NVIDIA NIM y Qdrant.
Material del profesor: `EAFIT-IA/si3003-artificial-intelligence` (notebooks de
`lecture10`).

> **Declaración de uso de IA generativa:** se utilizó un asistente de IA como
> apoyo en la preparación del repositorio y la redacción; la ejecución de los
> notebooks, las capturas y el análisis final son responsabilidad del autor.

## Estructura

```
Semana10-WorkshopRAG/
├── README.md                 ← este archivo (progreso e índice de evidencias)
├── RESPUESTAS.md             ← teoría, tablas, análisis y reflexión
├── notebooks/
│   ├── 02b_rag_app_nvidia_nim_colab.ipynb   ← app Gradio (A1–A4)
│   ├── 03_rag_workshop_papers_nvidia_nim_colab.ipynb  ← RAG papers (B1–B5, R1)
│   └── 04_challenges.ipynb   ← retos (C1, C4, C5, C6, filtro C3)
├── data/
│   └── company_documents.json   ← corpus propio (A2): empresa AeroNimbus
├── eval/
│   └── preguntas.csv         ← reto C4
└── evidencias/               ← capturas de Gradio (A1, A2, A4, C3)
```

## Progreso

### Núcleo (70 pts)

| Tarea | Descripción | Estado | Evidencia |
|---|---|---|---|
| T1 | Nodos por capa en HNSW | ✅ Completo | `RESPUESTAS.md` §T1 |
| T2 | ef, k y tamaño de la cola | ✅ Completo | `RESPUESTAS.md` §T2 |
| T3 | Inserción vs búsqueda | ✅ Completo | `RESPUESTAS.md` §T3 |
| T4 | Invariante de FAISSVectorStore | ✅ Completo | `RESPUESTAS.md` §T4 |
| T5 | Costo de la fuerza bruta | ✅ Completo | `RESPUESTAS.md` §T5 |
| A1 | Ejecutar app RAG de punta a punta | ⏳ Ejecutar en Colab | `evidencias/A1_gradio.png` |
| A2 | Corpus propio (AeroNimbus) | 🟡 Datos listos, falta ejecutar | `data/company_documents.json`, `evidencias/A2_corpus.png` |
| A3 | Ingeniería del prompt | 🟡 Prompt modificado, falta medir | `RESPUESTAS.md` §A3 |
| A4 | Score de similitud en fuentes | 🟡 Código listo, falta ejecutar | `evidencias/A4_scores.png` |
| B1 | RAG de papers + ground truth | ⏳ Ejecutar en Colab | `RESPUESTAS.md` §B1 |
| B2 | Experimentos chunk_size/k/threshold | 🟡 Funciones listas, falta correr | `RESPUESTAS.md` §B2 |
| B3 | Preguntas trampa | 🟡 Preguntas listas, falta correr | `RESPUESTAS.md` §B3 |
| B4 | Preguntas ES sobre papers EN | 🟡 Listo, falta correr | `RESPUESTAS.md` §B4 |
| B5 | Quiz propio con ground truth | 🟡 Celdas añadidas, falta correr | notebook 03 |
| R1 | Reflexión final | 🟡 Plantilla lista | `RESPUESTAS.md` §Reflexión |

### Retos (tope 30 pts) — opcionales

| Reto | Descripción | Estado |
|---|---|---|
| C4 | Evaluación automática de recuperación | 🟡 CSV + función listos |
| C6 | Persistencia del índice en disco | 🟡 Código listo |
| C5 | MMR vs similitud pura | 🟡 Código listo |
| C1 | HNSW vs fuerza bruta (FAISS) | 🟡 Código de arranque listo |
| C3 | App Gradio para papers con filtro | ⬜ Opcional |
| C2 | Tercer paper (HNSW) | ⬜ Opcional |

**Leyenda:** ✅ listo · 🟡 preparado, falta ejecutar/capturar en Colab · ⏳ requiere Colab · ⬜ no iniciado

## Cómo ejecutar en local (VS Code / Antigravity)

Ver **[`SETUP_LOCAL.md`](SETUP_LOCAL.md)** para la guía completa. Resumen:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env          # y pega tu NVIDIA_API_KEY dentro
```

Luego abre los notebooks de `notebooks/`, selecciona el kernel `.venv` y ejecuta
en orden: 02b → 03 → 04.

## Cómo ejecutar (en Google Colab)

1. Sube cada notebook de `notebooks/` a Colab.
2. En **🔑 Secrets** crea `NVIDIA_API_KEY` (gratis en https://build.nvidia.com/)
   y habilita el acceso para el notebook. **Nunca** escribas la key en una celda.
3. Para A2: sube `data/company_documents.json` a la raíz de la sesión de Colab
   (o ajusta la ruta — ver nota en el notebook 02b).
4. Ejecuta las celdas en orden y captura las pantallas pedidas en `evidencias/`.
5. Antes de entregar: busca `nvapi` en los notebooks y **borra las salidas con la
   key** si aparece.
