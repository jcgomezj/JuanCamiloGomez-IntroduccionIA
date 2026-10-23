# RESPUESTAS — Workshop RAG SI3003

Autor: **Juan Camilo Gómez** (`jcgomezj`)

> Nota: el **Bloque 0** (teoría) está completo porque no requiere ejecutar los
> modelos. Los bloques A, B, R y los retos tienen las tablas con plantilla: se
> completan con las salidas reales al ejecutar los notebooks en Colab. Las
> celdas marcadas con `[PENDIENTE]` esperan la evidencia de la ejecución.

---

# Bloque 0 · Teoría: HNSW y FAISS

## T1 · Cuántos nodos viven en cada capa

**Fórmula.** La capa máxima de un vector se sortea con
`l = ⌊ −ln(U) · mL ⌋` con `mL ≈ 1/ln(M)`. Con ese `mL`, la probabilidad de que
un vector llegue **al menos** a la capa `k` es `P(l ≥ k) = M^(−k)`.

### Cálculo a mano (M = 16, N = 1 000 000)

El número esperado en la capa `≥ k` es `N · M^(−k)`:

| Capa | Fórmula | Esperado (teórico) |
|---|---|---|
| ≥ 1 | 1 000 000 / 16 | **62 500** |
| ≥ 2 | 1 000 000 / 16² = /256 | **3 906,25 ≈ 3 906** |
| ≥ 3 | 1 000 000 / 16³ = /4096 | **244,14 ≈ 244** |

### Simulación en Python

Código ejecutado (equivalente a la pista; se usó la librería estándar para que
corra sin dependencias, pero el notebook `04_challenges.ipynb` incluye la
versión con NumPy de la pista):

```python
import random, math
M = 16
mL = 1 / math.log(M)
N = 1_000_000
random.seed(0)
counts = {1: 0, 2: 0, 3: 0}
for _ in range(N):
    u = random.random()
    l = math.floor(-math.log(1 - u) * mL)
    for k in (1, 2, 3):
        if l >= k:
            counts[k] += 1
for k in (1, 2, 3):
    print(f"capa >= {k}: simulado={counts[k]:>7,}  teorico={N / M**k:>12,.1f}")
```

**Salida de la simulación:**

```
capa >= 1: simulado= 62,253  teorico=     62,500.0
capa >= 2: simulado=  3,859  teorico=      3,906.2
capa >= 3: simulado=    234  teorico=        244.1
```

**Comparación.** Los valores simulados quedan a menos del 1 % del teórico en la
capa ≥ 1 y dentro del margen esperado por el ruido estadístico en las capas
altas (donde hay pocos nodos, la varianza relativa es mayor). La simulación
confirma el decaimiento exponencial `M^(−k)`.

**Por qué es bueno para la velocidad (2 frases).** Que existan pocos *hubs* en
las capas altas hace que el descenso inicial recorra un grafo diminuto y dé
saltos largos que acercan la búsqueda a la zona correcta en muy pocos pasos. Si
las capas altas estuvieran densamente pobladas, el descenso sería casi tan caro
como explorar toda la capa 0 y se perdería la ventaja logarítmica de HNSW.

---

## T2 · ef, k y el tamaño de la cola

1. **Si pides k = 10 con ef = 5, ¿qué falla y por qué?** Falla porque `ef` es el
   tamaño de la cola de candidatos en la capa 0, y la regla es `ef ≥ k`. Con
   `ef = 5` solo mantienes 5 candidatos finales, así que es **imposible
   devolver 10** vecinos: no hay suficientes candidatos para elegir. En la
   práctica devuelve a lo sumo 5 resultados (o de peor calidad).

2. **¿Con qué valor conviene empezar y cuándo se sube?** La presentación
   recomienda empezar con **`ef = 2·k`**. Si el *recall* resulta insuficiente,
   se **sube `ef`** (más candidatos → mejor recall, a costa de más latencia).
   Para RAG típico, `ef = 50–100` con `k = 5–10` ya da >95 % de recall.

3. **¿Cuál se cambia en producción sin reconstruir el índice?** El parámetro
   **`ef` (de búsqueda)**. `ef` y `ef_construction` son el mismo concepto —el
   tamaño de la cola de candidatos— pero en momentos distintos:
   `ef_construction` se fija al **construir** el índice y determina la calidad
   de las aristas, mientras que `ef` se puede ajustar **en cada consulta** sin
   tocar el grafo. Cambiar `ef_construction` obligaría a reconstruir el índice.

---

## T3 · Inserción frente a búsqueda

|  | Inserción | Búsqueda |
|---|---|---|
| **¿Se sortea la capa?** | **Sí**, con la distribución exponencial `l = ⌊−ln(U)·mL⌋`. | **No**; siempre se entra por la capa más alta del índice. |
| **Valor de ef en las capas altas** | `ef = 1` (greedy) al descender de `L_max` hasta `l+1`. | `ef = 1` (greedy) al descender de `L_max` hasta la capa 1. |
| **Parámetro de la fase principal** | `ef_construction` (capas `l` → 0). | `ef` de búsqueda (capa 0). |
| **¿Modifica el índice?** | **Sí**: crea aristas bidireccionales nuevas. | **No**: solo lee el grafo. |

**El porqué de cada fila:**
- *Sorteo:* en inserción hay que decidir en cuántas capas vivirá el vector; en
  búsqueda el query no se guarda, por eso no se sortea nada.
- *ef en capas altas:* en ambas operaciones el objetivo de las capas altas es
  solo aproximarse rápido a la zona, no precisión; por eso `ef = 1` greedy.
- *Fase principal:* al insertar interesa la **calidad de las conexiones**
  (`ef_construction`); al buscar interesa la **calidad de los resultados**
  (`ef`).
- *Modifica el índice:* insertar escribe aristas nuevas; buscar es de solo
  lectura.

**En el notebook 03:** la **inserción** se dispara con
`vectorstore.add_documents(...)` (Parte 3), y la **búsqueda** con
`retriever.invoke(...)` / `rag_chain.invoke(...)` (Partes 4 y 5).

---

## T4 · El invariante de FAISSVectorStore

El invariante es: **la posición `i` del índice FAISS corresponde siempre a
`self.documents[i]`.** FAISS solo habla de enteros; `self.documents` es lo único
que traduce esos enteros a documentos.

**Estado inicial: 4 chunks `[A, B, C, D]` en posiciones 0–3.**

| Posición FAISS | Antes de borrar | Después de borrar B (sin tocar FAISS) |
|---|---|---|
| 0 | A | A |
| 1 | B | C  ← desplazado |
| 2 | C | D  ← desplazado |
| 3 | D | (ya no existe en la lista) |

Alguien elimina `B` de `self.documents` pero **no** del índice FAISS. La lista
pasa a `[A, C, D]` (posiciones 0,1,2), mientras que FAISS sigue teniendo 4
vectores.

**Una búsqueda que antes devolvía los índices `[1, 2, 3]`:**

| Índice FAISS devuelto | Lo que la búsqueda *cree* que es | Lo que `self.documents[i]` devuelve ahora | Resultado |
|---|---|---|---|
| 1 | B | **C** | ❌ error silencioso (documento equivocado, sin excepción) |
| 2 | C | **D** | ❌ error silencioso |
| 3 | D | **IndexError** (la lista solo tiene índices 0–2) | 💥 excepción |

**¿Cuál es error silencioso y cuál lanza excepción?**
- Índices **1 y 2** → **error silencioso**: devuelven C y D en vez de B y C. El
  sistema responde con confianza una respuesta equivocada, sin ninguna señal de
  error. Es el caso más peligroso.
- Índice **3** → **excepción** (`IndexError`): FAISS pide la posición 3 pero la
  lista ahora solo llega hasta la 2.

**Relación con bases vectoriales (Qdrant, Weaviate).** `faiss.IndexFlatIP` no
tiene operación de borrado: eliminar un documento desalinea el mapeo y hay que
reconstruir el índice completo. Precisamente por eso existen las bases
vectoriales de producción como **Qdrant**: gestionan inserciones, borrados y
actualizaciones de forma nativa, manteniendo la consistencia entre el vector y
su documento sin reconstruir todo el índice.

---

## T5 · El costo de la fuerza bruta

Escenario: 1 000 000 de vectores de 768 dimensiones, ~768 millones de
multiplicaciones por consulta, 100 consultas por segundo.

**Cuenta:**

```
7,68 × 10^8 multiplicaciones/consulta  ×  100 consultas/segundo
= 7,68 × 10^10 multiplicaciones/segundo
= 76 800 000 000 multiplicaciones por segundo
```

**En notación científica: `7,68 × 10^10 mult/s`.**

**Por qué "ser exacto" deja de ser viable.** Una búsqueda exacta (fuerza bruta)
compara el query contra **todos** los vectores: el costo crece linealmente con
`N` y con la dimensión, y a escala real (millones de vectores, cientos de
consultas por segundo) llega a decenas de miles de millones de operaciones por
segundo, lo que exige hardware costoso y añade latencia inaceptable para una
aplicación interactiva.

**Qué se sacrifica y qué se gana en ANN (1 frase).** En ANN se sacrifica la
garantía de encontrar siempre al vecino exacto (se acepta un *recall* < 100 %) a
cambio de ganar búsquedas varios órdenes de magnitud más rápidas gracias a
estructuras como HNSW.

---

# Bloque A · App RAG empresarial (Notebook 02b)

> Ejecutar `notebooks/02b_rag_app_nvidia_nim_colab.ipynb` en Colab con la
> `NVIDIA_API_KEY` en Secrets. El corpus propio está en
> `data/company_documents.json` (empresa inventada: **AeroNimbus**,
> monitoreo de drones).

## A1 · Ejecuta la app RAG de punta a punta

Tabla de 5 preguntas con el corpus de **demostración** del notebook (las 4 de
ejemplo + 1 fuera de contexto):

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas |
|---|---|---|---|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | `[PENDIENTE]` | `[PENDIENTE]` |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | `[PENDIENTE]` | `[PENDIENTE]` |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | `[PENDIENTE]` | `[PENDIENTE]` |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | `[PENDIENTE]` | `[PENDIENTE]` |
| 5 | (fuera de contexto) ¿Cuál es la capital de Francia? | `[PENDIENTE: debería decir que no tiene esa información]` | `[PENDIENTE]` |

Evidencia: `evidencias/A1_gradio.png`.

## A2 · Tu propio corpus de documentos

Corpus propio: **AeroNimbus** (plataforma de monitoreo de drones), 10 documentos
en `data/company_documents.json`, cada uno con metadata `department` y `type`.

5 preguntas que SÍ se responden + 2 que NO:

| # | Pregunta | Tipo | Respuesta del RAG | Fuentes | ¿Correcta? |
|---|---|---|---|---|---|
| 1 | ¿Cuánto cuesta el plan Fleet de AeroNimbus? | responde | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 2 | ¿Quién aprueba un descuento del 25 %? | responde | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 3 | ¿En cuánto tiempo responde el soporte prioritario un incidente crítico? | responde | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 4 | ¿Cuántos días por semana se puede trabajar remoto? | responde | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 5 | ¿Cuánto tiempo se conservan los datos de vuelo en el plan Enterprise? | responde | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 6 | ¿Qué marca de baterías usan los drones? | NO responde | `[PENDIENTE: debería decir que no sabe]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 7 | ¿Cuál es la política de vacaciones? | NO responde | `[PENDIENTE: debería decir que no sabe]` | `[PENDIENTE]` | `[PENDIENTE]` |

**Ajuste de `chunk_size`:** mis documentos son relativamente cortos (2–4 frases
cada uno). Con `chunk_size=400` cada documento cabe casi en un solo chunk, lo
que conserva el contexto completo de cada política/spec. `[COMPLETAR tras
ejecutar: si subo o bajo chunk_size y por qué]`.

Evidencia: `evidencias/A2_corpus.png`.

## A3 · Ingeniería del prompt (antes y después)

**Diff del system prompt** (ver el notebook, celda `build_pipeline`):

```diff
-     "Eres un asistente empresarial. Responde únicamente con información "
-     "del contexto proporcionado. Si el contexto no contiene la respuesta, "
-     "indica que no tienes esa información. Responde en español y de forma concisa.",
+     "Eres un asistente empresarial. Responde únicamente con información "
+     "del contexto proporcionado. Cita siempre el título del documento entre "
+     "corchetes, por ejemplo [Política de reembolsos a clientes]. Si el contexto "
+     "no alcanza, responde exactamente: «No encuentro esa información en los "
+     "documentos.» Responde en español, en máximo 3 oraciones.",
```

Tabla comparativa (1 fácil, 1 ambigua, 1 fuera de contexto):

| Pregunta | Tipo | Antes | Después |
|---|---|---|---|
| ¿Cuánto cuesta el plan Fleet? | fácil | `[PENDIENTE]` | `[PENDIENTE]` |
| ¿Qué plan me conviene? | ambigua | `[PENDIENTE]` | `[PENDIENTE]` |
| ¿Qué marca de baterías usan? | fuera de contexto | `[PENDIENTE]` | `[PENDIENTE]` |

**Conclusión** `[COMPLETAR]`: qué instrucción tuvo más efecto (p. ej. la frase
fija para "no sé") y cuál el modelo tendió a ignorar (p. ej. el límite de 3
oraciones o el formato de citas).

## A4 · Muestra el score de similitud en las fuentes

Tabla de scores (3 buenas + 2 fuera de contexto):

| # | Pregunta | Tipo | Fuente 1 (score) | Fuente 2 (score) | Fuente 3 (score) |
|---|---|---|---|---|---|
| 1 | ¿Cuánto cuesta el plan Fleet? | buena | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 2 | ¿Quién aprueba un descuento del 25 %? | buena | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 3 | ¿Cuánto se conservan los datos en Enterprise? | buena | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 4 | ¿Qué marca de baterías usan? | fuera | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| 5 | ¿Cuál es la política de vacaciones? | fuera | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |

**Umbral propuesto** `[COMPLETAR]`: a partir de la tabla, los scores de las
preguntas buenas rondan `[X]` y los de las fuera de contexto `[Y]`; propongo
`score_threshold = [Z]` porque separa ambos grupos sin descartar respuestas
válidas.

Evidencia: `evidencias/A4_scores.png`.

---

# Bloque B · RAG sobre papers académicos (Notebook 03)

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

| # | Pregunta (resumen) | Respuesta del RAG (resumen) | Correcta | Paper correcto |
|---|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 En→De | `[PENDIENTE]` (GT: 28.4) | `[sí/parcial/no]` | `[sí/no]` |
| Q2 | Cómo combina RAG memoria paramétrica y no paramétrica | `[PENDIENTE]` | `[ ]` | `[ ]` |
| Q3 | Propósito del factor de escala 1/√dk | `[PENDIENTE]` | `[ ]` | `[ ]` |
| Q4 | Las dos variantes de RAG (Sequence vs Token) | `[PENDIENTE]` | `[ ]` | `[ ]` |
| Q5 | Nº de attention heads y dims por cabeza (8, dk=dv=64) | `[PENDIENTE]` | `[ ]` | `[ ]` |
| Bonus | Generador de RAG (BART) y si es Transformer | `[PENDIENTE]` | `[ ]` | `[ambos papers? sí/no]` |

**Bonus — fragmentos recuperados** `[COMPLETAR: listar títulos/archivos y decir
si aparecieron los dos papers]`.

## B2 · Experimentos con chunk_size, k y score_threshold

> La celda con las funciones auxiliares (`build_store`, `build_chain`) ya está
> añadida al notebook después de la Parte 4.

| # | Experimento | Parámetro | n chunks | Aciertos (de 5) | Observación |
|---|---|---|---|---|---|
| 1 | chunk_size | 400 (overlap≈52) | `[ ]` | `[ ]` | `[ ]` |
| 2 | chunk_size | 900 (overlap≈120) | `[ ]` | `[ ]` | `[ ]` |
| 3 | chunk_size | 1500 (overlap≈195) | `[ ]` | `[ ]` | `[ ]` |
| 4 | k (índice 900) | k = 3 | `[ ]` | `[ ]` | `[ ]` |
| 5 | k (índice 900) | k = 5 | `[ ]` | `[ ]` | `[ ]` |
| 6 | k (índice 900) | k = 8 | `[ ]` | `[ ]` | `[ ]` |
| 7 | threshold (índice 900) | 0.15 | `[ ]` | `[ ]` | `[ ]` |
| 8 | threshold (índice 900) | 0.25 | `[ ]` | `[ ]` | `[ ]` |
| 9 | threshold (índice 900) | 0.40 | `[ ]` | `[ ]` | `[ ]` |

**Conclusión (5–8 líneas)** `[COMPLETAR]`: qué parámetro movió más los
resultados, si hay una configuración mejor que la del notebook (900/120/5/0.25).

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

| # | Pregunta trampa (no está en los papers) | Clasificación | Notas |
|---|---|---|---|
| 1 | What BLEU score did GPT-4 achieve on WMT 2022? | `[admite / inventa / mezcla]` | `[ ]` |
| 2 | How many parameters does the Transformer-XXL from 2023 have? | `[ ]` | `[ ]` |
| 3 | Which optimizer does the RAG paper recommend for LLaMA fine-tuning? | `[ ]` | `[ ]` |

**Diff del prompt (si hubo que corregir una alucinación)** `[COMPLETAR si
aplica]`.

## B4 · Preguntas en español sobre papers en inglés

Traducción de Q1, Q3 y Q4 al español, comparando recuperación EN vs ES:

| Pregunta | Chunks recuperados (EN) | Chunks recuperados (ES) | ¿Mismos chunks? | Calidad respuesta |
|---|---|---|---|---|
| Q1 (BLEU Transformer big) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Q3 (factor de escala √dk) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Q4 (variantes de RAG) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Hipótesis** `[COMPLETAR]`: el modelo `nemotron-3-embed-1b` parece/no parece
alinear español e inglés en el mismo espacio semántico; cómo lo comprobaría
(p. ej. midiendo overlap de chunks EN/ES o la similitud coseno entre la query en
ambos idiomas).

## B5 · Escribe tu propio quiz con ground truth

> 3 preguntas nuevas añadidas al notebook como celdas markdown (bloque
> `<details>` con el ground truth) + celdas de código. Una exige combinar ambos
> papers.

| # | Pregunta nueva | Ground truth (sección) | Respuesta del RAG | Correcta |
|---|---|---|---|---|
| N1 | `[tu pregunta sobre Attention]` | `[sección]` | `[PENDIENTE]` | `[ ]` |
| N2 | `[tu pregunta sobre RAG]` | `[sección]` | `[PENDIENTE]` | `[ ]` |
| N3 | `[pregunta que cruza ambos papers]` | `[secciones]` | `[PENDIENTE]` | `[ ]` |

---

# Reflexión

> R1 — Notebook 03. Cada respuesta ≥ 3 oraciones, apoyada en una tabla, un score
> o un experimento de hoy. Completar tras ejecutar los notebooks.

1. **¿Qué ocurrió cuando el contexto no contenía información suficiente?**
   `[COMPLETAR, ≥3 oraciones, apóyate en B3]`

2. **¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos
   papers?** `[COMPLETAR, apóyate en la salida de B1 bonus]`

3. **¿Qué efecto tuvo modificar chunk_size, chunk_overlap, k o
   score_threshold?** `[COMPLETAR, apóyate en la tabla de B2]`

4. **¿Por qué conviene separar el modelo de embeddings del modelo generativo?**
   `[COMPLETAR: especialización, costo, posibilidad de cambiar uno sin el otro,
   dimensionalidad del índice]`

5. **¿Qué cambiarías para llevar este prototipo a producción?**
   `[COMPLETAR: persistencia (Qdrant en disco, ver C6), control de acceso,
   evaluación automática (ver C4), observabilidad/logging, costos de embeddings
   y LLM]`

---

# Retos opcionales

> Marca abajo solo los que hagas. El código de arranque de C1, C4, C5, C6 y el
> filtro de C3 están en `notebooks/04_challenges.ipynb`.

## C4 · Evaluación automática de la recuperación

Archivo `eval/preguntas.csv` creado con 10 preguntas (`pregunta`,
`fuente_esperada`, `palabra_clave`). Función `hit_at_k` en el notebook.

| chunk_size | hit@3 | hit@5 |
|---|---|---|
| 400 | `[PENDIENTE]` | `[PENDIENTE]` |
| 900 | `[PENDIENTE]` | `[PENDIENTE]` |
| 1500 | `[PENDIENTE]` | `[PENDIENTE]` |

**Dos limitaciones de esta métrica** `[COMPLETAR]`: (1) depende de que la
palabra clave aparezca literalmente, no mide comprensión semántica; (2) no
evalúa si el *orden* de los chunks es bueno ni si la respuesta final del LLM fue
correcta, solo la recuperación.
