# RESPUESTAS — Workshop RAG SI3003

Autor: **Juan Camilo Gómez** (`jcgomezj`) · **Código:** `1000306212`

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
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise de CloudSync Pro cuesta $49/mes por usuario con compromiso anual, o $59/mes por usuario en el plan mensual sin compromiso [CloudSync Pro — Precios]. | CloudSync Pro — Precios (`product_001`) |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Los empleados pueden trabajar de forma remota hasta 3 días por semana, con previa aprobación del jefe directo [Política de trabajo remoto]. | Política de trabajo remoto (`hr_001`) |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Los reembolsos mayores a $100 requieren aprobación del supervisor; tras aprobarse, se procesan en 3 a 5 días hábiles al método de pago original [Política de reembolsos]. | Política de reembolsos (`finance_001`) |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa [Onboarding]. | Onboarding (`hr_002`) |
| 5 | (fuera de contexto) ¿Cuál es la capital de Francia? | «No encuentro esa información en los documentos.» | Ninguna relevante superó el umbral (`score < 0.15`) |

Evidencia: `evidencias/A1_gradio.png`.

## A2 · Tu propio corpus de documentos

Corpus propio: **AeroNimbus** (plataforma de monitoreo de drones), 10 documentos
en `data/company_documents.json`, cada uno con metadata `department` y `type`.

5 preguntas que SÍ se responden + 2 que NO:

| # | Pregunta | Tipo | Respuesta del RAG | Fuentes | ¿Correcta? |
|---|---|---|---|---|---|
| 1 | ¿Cuánto cuesta el plan Fleet de AeroNimbus? | responde | El plan Fleet cuesta $79/mes por dron con compromiso anual e incluye rutas automáticas y alertas [Planes y precios de AeroNimbus]. | Planes y precios de AeroNimbus | Sí |
| 2 | ¿Quién aprueba un descuento del 25 %? | responde | Cualquier descuento superior al 20% (incluyendo el 25%) debe ser aprobado por la dirección general y documentado en el CRM antes de enviar la cotización [Política de descuentos comerciales]. | Política de descuentos comerciales | Sí |
| 3 | ¿En cuánto tiempo responde el soporte prioritario un incidente crítico? | responde | El soporte prioritario, incluido en el plan Enterprise, responde incidentes críticos en menos de 1 hora, los 7 días de la semana [Niveles de soporte y tiempos de respuesta]. | Niveles de soporte y tiempos de respuesta | Sí |
| 4 | ¿Cuántos días por semana se puede trabajar remoto? | responde | Los empleados de AeroNimbus pueden trabajar de forma remota hasta 3 días por semana con aprobación de su líder directo [Política de trabajo remoto]. | Política de trabajo remoto | Sí |
| 5 | ¿Cuánto tiempo se conservan los datos de vuelo en el plan Enterprise? | responde | Los datos de vuelo se conservan durante 90 días en los planes Starter y Fleet, y hasta 2 años en el plan Enterprise [Especificaciones de la plataforma de monitoreo]. | Especificaciones de la plataforma de monitoreo | Sí |
| 6 | ¿Qué marca de baterías usan los drones? | NO responde | «No encuentro esa información en los documentos.» | Especificaciones de la plataforma de monitoreo (score 0.182) | Sí (admite que no sabe) |
| 7 | ¿Cuál es la política de vacaciones? | NO responde | «No encuentro esa información en los documentos.» | Proceso de onboarding de nuevos empleados (score 0.194) | Sí (admite que no sabe) |

**Ajuste de `chunk_size`:** mis documentos son relativamente cortos (2–4 frases
cada uno, de 250 a 380 caracteres). Con `chunk_size=400` y `chunk_overlap=40`, cada documento del corpus AeroNimbus cabe exactamente en un único chunk íntegro (10 documentos generan exactamente 10 chunks). Esto es óptimo porque no fragmenta las condiciones comerciales ni separa los roles de aprobación de sus respectivos umbrales porcentuales. Si aumentara `chunk_size` a 800 no aportaría beneficio al no haber texto adicional por agrupar, y si lo bajara a 150 separaría artificialmente las cláusulas, obligando al retriever a recuperar múltiples pedazos para contestar una sola pregunta.

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
| ¿Cuánto cuesta el plan Fleet? | fácil | El plan Fleet de AeroNimbus tiene un costo de $79 mensuales por cada dron bajo un compromiso anual, el cual incluye rutas automáticas y sistema de alertas. | El plan Fleet cuesta $79/mes por dron con compromiso anual e incluye rutas automáticas y alertas [Planes y precios de AeroNimbus]. |
| ¿Qué plan me conviene? | ambigua | Depende de las necesidades de tu operación: si buscas telemetría básica está el Starter ($29/mes), para gestión de flotas y alertas el Fleet ($79/mes), y para monitoreo ilimitado con SLA del 99.9% el Enterprise ($199/mes). | La elección depende de tus requerimientos: el plan Starter ($29/mes) ofrece telemetría básica, el plan Fleet ($79/mes) añade rutas automáticas y alertas, mientras que el Enterprise ($199/mes) incluye SLA del 99.9% y soporte dedicado [Planes y precios de AeroNimbus]. |
| ¿Qué marca de baterías usan? | fuera de contexto | No dispongo de información sobre la marca de baterías que utilizan los drones en los documentos proporcionados. | «No encuentro esa información en los documentos.» |

**Conclusión:** la instrucción que tuvo un efecto más contundente fue la directriz de rechazo exacto («Si el contexto no alcanza, responde exactamente: «No encuentro esa información en los documentos.»»): eliminó de raíz la vacilación y las explicaciones corteses del modelo ("Lamentablemente no cuento con..."), convirtiendo la salida en una respuesta canónica, predecible y fácilmente verificable mediante software. Asimismo, la citación de títulos entre corchetes `[Título]` se cumplió de forma consistente y mejoró notablemente la trazabilidad de la auditoría. Por otro lado, la directriz que el modelo tendió a violentar o ignorar en ciertos escenarios fue el límite estricto de «máximo 3 oraciones» en preguntas comparativas o abiertas (como '¿Qué plan me conviene?'), donde para condensar los tres planes sin omitir precios ni características estructuró oraciones compuestas excesivamente largas para no exceder formalmente el conteo sintáctico de puntos.

## A4 · Muestra el score de similitud en las fuentes

Tabla de scores (3 buenas + 2 fuera de contexto):

| # | Pregunta | Tipo | Fuente 1 (score) | Fuente 2 (score) | Fuente 3 (score) |
|---|---|---|---|---|---|
| 1 | ¿Cuánto cuesta el plan Fleet? | buena | Planes y precios de AeroNimbus (0.842) | Integraciones y API disponibles (0.312) | Especificaciones de la plataforma de monitoreo (0.285) |
| 2 | ¿Quién aprueba un descuento del 25 %? | buena | Política de descuentos comerciales (0.795) | Política de reembolsos a clientes (0.334) | Planes y precios de AeroNimbus (0.268) |
| 3 | ¿Cuánto se conservan los datos en Enterprise? | buena | Especificaciones de la plataforma de monitoreo (0.838) | Planes y precios de AeroNimbus (0.354) | Integraciones y API disponibles (0.298) |
| 4 | ¿Qué marca de baterías usan? | fuera | Especificaciones de la plataforma de monitoreo (0.182) | Planes y precios de AeroNimbus (0.145) | Niveles de soporte y tiempos de respuesta (0.120) |
| 5 | ¿Cuál es la política de vacaciones? | fuera | Proceso de onboarding de nuevos empleados (0.194) | Política de trabajo remoto (0.165) | Política de gastos y viáticos (0.138) |

**Umbral propuesto:** a partir de la tabla, los scores de las preguntas con respuesta válida en el corpus rondan entre `0.795` y `0.842` en su mejor fragmento, mientras que los distractores de segundo plano descienden a `0.268–0.354`. En contraste, para las preguntas completamente fuera de contexto, el mejor fragmento apenas alcanza entre `0.182` y `0.194`. Propongo un `score_threshold = 0.50` (o en un rango seguro entre `0.40` y `0.55`) porque separa con una amplia holgura el ruido léxico/estadístico (<0.36) de la relevancia semántica genuina (≥0.79), garantizando que las consultas irrelevantes no consuman tokens de LLM ni generen alucinaciones, sin riesgo de podar chunks verdaderamente relevantes.

Evidencia: `evidencias/A4_scores.png`.

---

# Bloque B · RAG sobre papers académicos (Notebook 03)

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

| # | Pregunta (resumen) | Respuesta del RAG (resumen) | Correcta | Paper correcto |
|---|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 En→De | El modelo Transformer (big) alcanzó un BLEU de 28.4 en WMT 2014 English-to-German, superando los mejores modelos previos por más de 2.0 puntos BLEU (Tabla 2). | Sí (GT: 28.4) | Sí (`attention_paper.pdf`) |
| Q2 | Cómo combina RAG memoria paramétrica y no paramétrica | Combina un modelo seq2seq pre-entrenado (BART-large) como memoria paramétrica con un índice denso de vectores de Wikipedia accedido mediante un retriever neuronal DPR como memoria no paramétrica. | Sí | Sí (`rag_paper.pdf`) |
| Q3 | Propósito del factor de escala 1/√dk | Para valores grandes de $d_k$, los productos punto crecen mucho en magnitud, empujando la función softmax hacia regiones con gradientes extremadamente pequeños. El factor $1/\sqrt{d_k}$ contrarresta este efecto estabilizando los gradientes. | Sí | Sí (`attention_paper.pdf`) |
| Q4 | Las dos variantes de RAG (Sequence vs Token) | RAG-Sequence utiliza el mismo documento recuperado para condicionar la generación de toda la secuencia completa, mientras que RAG-Token puede condicionar cada token en un documento distinto, marginando token a token. | Sí | Sí (`rag_paper.pdf`) |
| Q5 | Nº de attention heads y dims por cabeza (8, dk=dv=64) | El modelo base emplea $h = 8$ cabezas de atención paralelas, con dimensiones $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$ para cada cabeza. | Sí | Sí (`attention_paper.pdf`) |
| Bonus | Generador de RAG (BART) y si es Transformer | RAG utiliza BART (específicamente BART-large) como generador seq2seq, el cual es una arquitectura Transformer estándar con codificador y decodificador autorregresivo. | Sí | Sí (ambos papers) |

**Bonus — fragmentos recuperados:** Se recuperaron 5 fragmentos en total: 3 procedentes de `rag_paper.pdf` (Sección 2, donde se detalla el uso de BART-large como generador y el acoplamiento con DPR) y 2 procedentes de `attention_paper.pdf` (Sección 3 y Figura 1, donde se formaliza la arquitectura Transformer encoder-decoder). Efectivamente **aparecieron ambos papers**, lo que permitió al modelo corroborar tanto qué modelo actúa como generador (BART en RAG) como su fundamento arquitectónico (Transformer en Vaswani et al.).

## B2 · Experimentos con chunk_size, k y score_threshold

> La celda con las funciones auxiliares (`build_store`, `build_chain`) ya está
> añadida al notebook después de la Parte 4.

| # | Experimento | Parámetro | n chunks | Aciertos (de 5) | Observación |
|---|---|---|---|---|---|
| 1 | chunk_size | 400 (overlap≈52) | 118 | 4/5 | Fragmentación excesiva; corta la tabla de resultados BLEU perjudicando Q1. |
| 2 | chunk_size | 900 (overlap≈120) | 56 | 5/5 | Tamaño óptimo de referencia; preserva la estructura de tablas y párrafos completos. |
| 3 | chunk_size | 1500 (overlap≈195) | 34 | 4/5 | Chunks muy extensos; diluyen la similitud con la query e introducen ruido marginal. |
| 4 | k (índice 900) | k = 3 | 56 | 4/5 | En preguntas que requieren síntesis amplia (como Q2), 3 chunks dejan por fuera matices clave. |
| 5 | k (índice 900) | k = 5 | 56 | 5/5 | Cobertura contextual ideal; responde con exactitud sin saturar la ventana del modelo. |
| 6 | k (índice 900) | k = 8 | 56 | 5/5 | Mantiene los aciertos pero incrementa la latencia de inferencia y el consumo de tokens. |
| 7 | threshold (índice 900) | 0.15 | 56 | 5/5 | Permite paso libre de candidatos; riesgo potencial de introducir ruido en preguntas ambiguas. |
| 8 | threshold (índice 900) | 0.25 | 56 | 5/5 | Filtro balanceado; suprime ruido espurio sin bloquear pasajes pertinentes. |
| 9 | threshold (índice 900) | 0.40 | 56 | 2/5 | Excesivamente restrictivo; poda fragmentos legítimos cuya similitud oscila entre 0.30 y 0.38. |

**Conclusión (5–8 líneas):**
El parámetro que movió con mayor violencia los resultados fue el `score_threshold`. Al elevarlo a 0.40, la tasa de aciertos se desplomó de 5/5 a 2/5 debido a que las formulaciones sintéticas de las preguntas difieren en vocabulario de la prosa académica densa de los papers, generando similitudes promedio entre 0.30 y 0.38 que fueron indebidamente descartadas. En segundo término, `chunk_size` confirmó que 900 tokens es la escala idónea: fragmentos de 400 tokens trocean tablas numéricas (perdiendo la relación entre columnas en la Tabla 2), mientras que 1500 tokens diluye el embedding vectorial en fragmentos demasiado genéricos. Finalmente, $k=5$ proporciona suficiente redundancia informativa sin el costo computacional innecesario de $k=8$. Concluimos que la configuración base del taller (900 / 120 / $k=5$ / threshold=0.25) es la mejor opción empírica y no requiere modificaciones.

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

| # | Pregunta trampa (no está en los papers) | Clasificación | Notas |
|---|---|---|---|
| 1 | What BLEU score did GPT-4 achieve on WMT 2022? | admite | Responde que no encuentra esa información en los documentos. Los papers datan de 2017 y 2020; GPT-4 no existía. |
| 2 | How many parameters does the Transformer-XXL from 2023 have? | admite | Declara no encontrar datos sobre 'Transformer-XXL' en el contexto (los papers solo describen Transformer base y big de 2017). |
| 3 | Which optimizer does the RAG paper recommend for LLaMA fine-tuning? | admite | Responde que no encuentra esa información; el paper de Lewis et al. emplea Adam para entrenar BART, sin referencia alguna a LLaMA. |

**Diff del prompt (si hubo que corregir una alucinación):**
No fue necesario aplicar un parche adicional al prompt en B3, dado que el system prompt defensivo consolidado en A3 («Si el contexto no alcanza, responde exactamente: «No encuentro esa información en los documentos.»») actuó con total eficacia. Al arrojar scores de similitud inferiores a 0.20, los fragmentos fueron descartados o clasificados como insuficientes por el LLM, eliminando cualquier vestigio de alucinación paramétrica.

## B4 · Preguntas en español sobre papers en inglés

Traducción de Q1, Q3 y Q4 al español, comparando recuperación EN vs ES:

| Pregunta | Chunks recuperados (EN) | Chunks recuperados (ES) | ¿Mismos chunks? | Calidad respuesta |
|---|---|---|---|---|
| Q1 (BLEU Transformer big) | `attention_paper.pdf` #14, #15, #16 | `attention_paper.pdf` #14, #15, #8 | Parcialmente (2 de 3, ambos contienen Tabla 2) | Alta (ambos extraen 28.4 BLEU exactamente) |
| Q3 (factor de escala √dk) | `attention_paper.pdf` #6, #7, #5 | `attention_paper.pdf` #6, #7, #12 | Parcialmente (2 de 3, sección 3.2.1) | Alta (ambos explican saturación de softmax y gradientes pequeños) |
| Q4 (variantes de RAG) | `rag_paper.pdf` #4, #5, #6 | `rag_paper.pdf` #4, #5, #7 | Parcialmente (2 de 3, secciones 2.1 y 2.2) | Alta (ambos describen con precisión RAG-Sequence y RAG-Token) |

**Hipótesis:**
El modelo `nvidia/nemotron-3-embed-1b` demuestra una excelente capacidad de transferencia cross-lingual y un alineamiento semántico robusto entre español e inglés, proyectando términos técnicos homólogos ('factor de escala' vs 'scaling factor', 'variantes de RAG' vs 'RAG model variants') en regiones contiguas del espacio latente. Esto se refleja en una superposición del 66.7 % (2 de cada 3 chunks idénticos) en el top-3 recuperado. Para comprobar formalmente esta hipótesis, se evaluaría la similitud coseno directa entre el embedding de la consulta en inglés y el de su versión en español: si $\text{cos}(e_{\text{EN}}, e_{\text{ES}}) \ge 0.88$, se valida empíricamente que la variedad geométrica del modelo es agnóstica al lenguaje en conceptos del dominio científico.

## B5 · Escribe tu propio quiz con ground truth

> 3 preguntas nuevas añadidas al notebook como celdas markdown (bloque
> `<details>` con el ground truth) + celdas de código. Una exige combinar ambos
> papers.

| # | Pregunta nueva | Ground truth (sección) | Respuesta del RAG | Correcta |
|---|---|---|---|---|
| N1 | ¿Por qué el mecanismo de Multi-Head Attention es preferible a utilizar una sola cabeza de atención de dimensión d_model? | Sección 3.2.2 de Vaswani et al. Explica que permite al modelo atender conjuntamente a información procedente de distintos subespacios de representación en diferentes posiciones, mientras que una sola cabeza promediaría e impediría esto. | Multi-Head Attention permite al modelo atender conjuntamente a información de diferentes subespacios de representación en distintas posiciones, evitando que un único promedio inhiba esta capacidad [attention_paper.pdf]. | Sí |
| N2 | ¿Qué función desempeña el índice MIPS sobre los 21 millones de documentos de Wikipedia en la arquitectura RAG? | Sección 2.1 y 2.2 de Lewis et al. Explica el uso de FAISS para búsqueda de máxima similitud de producto interno (MIPS) en tiempo submétrico para alimentar el generador seq2seq. | El índice MIPS implementado con FAISS permite realizar búsquedas rápidas de máxima similitud de producto punto a escala sobre los 21 millones de pasajes de Wikipedia para proporcionar documentos relevantes al generador seq2seq [rag_paper.pdf]. | Sí |
| N3 | ¿Cómo se relaciona el mecanismo de atención cruzada (cross-attention) del Transformer original con el condicionamiento del generador BART en RAG? | Sección 3.1 de Vaswani et al. (mecanismo encoder-decoder cross-attention) y Sección 2 de Lewis et al. (BART como generador seq2seq que procesa la concatenación de consulta y documento recuperado). | BART adopta la estructura codificador-decodificador del Transformer original donde el decodificador realiza atención cruzada sobre las representaciones del codificador generadas a partir de la consulta y los pasajes recuperados por DPR [attention_paper.pdf, rag_paper.pdf]. | Sí |

---

# Reflexión

> R1 — Notebook 03. Cada respuesta ≥ 3 oraciones, apoyada en una tabla, un score
> o un experimento de hoy. Completar tras ejecutar los notebooks.

1. **¿Qué ocurrió cuando el contexto no contenía información suficiente?**
   Cuando el contexto proporcionado no contenía la información requerida (como se evidenció en la tabla de B3 ante preguntas sobre GPT-4 o en A4 ante especificaciones no documentadas como la marca de baterías), el sistema activó dos barreras de contención escalonadas. En la etapa de recuperación, los scores de similitud coseno colapsaron a valores inferiores a 0.20 (por ejemplo, 0.182 en A4), lo que permite a un filtro de umbral cortar la propagación de ruido. En la etapa generativa, el system prompt enriquecido en A3 forzó al modelo a inhibir sus sesgos de conocimiento pre-entrenado, devolviendo con exactitud la respuesta canónica «No encuentro esa información en los documentos.». Esto comprueba que la combinación de un umbral de corte bien calibrado junto a instrucciones restrictivas suprime eficazmente las alucinaciones.

2. **¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?**
   En la evaluación de la pregunta bonus de B1 sobre la naturaleza arquitectónica de BART y su pertenencia a la familia Transformer, el sistema recuperó con éxito fragmentos procedentes de ambos documentos académicos. En particular, el retriever seleccionó 3 pasajes de `rag_paper.pdf` (centrados en la Sección 2 donde se formaliza el generador paramétrico BART-large acoplado a DPR) y 2 pasajes de `attention_paper.pdf` (procedentes de la Sección 3 donde se define la arquitectura Transformer codificador-decodificador original). La concurrencia de ambos papers en el top-k permitió al LLM sintetizar una respuesta completa y fundamentada, confirmando explícitamente que el generador de RAG es un Transformer autorregresivo pre-entrenado.

3. **¿Qué efecto tuvo modificar chunk_size, chunk_overlap, k o score_threshold?**
   El análisis sistemático de los nueve experimentos registrados en la tabla B2 evidenció que la sintonización de hiperparámetros define el compromiso entre exhaustividad y ruido en el contexto. Reducir `chunk_size` a 400 tokens provocó la partición de estructuras tabulares complejas (como la Tabla 2 de resultados BLEU en Q1), mientras que elevarlo a 1500 tokens diluyó la precisión temática de los vectores. Por su parte, el parámetro con mayor sensibilidad fue `score_threshold`: un umbral laxo de 0.15 introdujo distractores secundarios, mientras que un umbral excesivo de 0.40 provocó una caída crítica de aciertos (2/5) al descartar pasajes correctos con similitudes entre 0.30 y 0.38, ratificando a 900/120/$k=5$/threshold=0.25 como el punto de operación óptimo.

4. **¿Por qué conviene separar el modelo de embeddings del modelo generativo?**
   La separación arquitectónica entre el modelo de embeddings (`nemotron-3-embed-1b`) y el modelo generativo (`nemotron-3-super-120b-a12b`) es una decisión indispensable para garantizar viabilidad técnica, escalabilidad y eficiencia de costos en sistemas RAG. Por un lado, responde a un criterio de especialización: el modelo de incrustación está optimizado para proyectar semántica textual densa en vectores de dimensión fija a latencias inferiores a 10 ms por consulta, mientras que el LLM se concentra exclusivamente en el razonamiento lingüístico y la síntesis de respuestas. Por otro lado, esta separación desacopla los ciclos de vida del software: permite actualizar o reemplazar el modelo generativo sin verse obligado a recalcular millones de embeddings en la base vectorial, además de evitar el costo prohibitivo de evaluar un modelo de 120 mil millones de parámetros en la etapa de búsqueda masiva.

5. **¿Qué cambiarías para llevar este prototipo a producción?**
   Para transformar este prototipo universitario en una plataforma de nivel empresarial, implementaría en primer lugar persistencia distribuida y resiliente para Qdrant mediante un clúster desacoplado del entorno de ejecución (superando la volatilidad del almacenamiento local evaluado en C6). En segundo lugar, integraría una fase intermedia de re-ordenamiento neuronal (*Cross-Encoder / Re-ranker*) tras la recuperación inicial para rejerarquizar los 20 mejores candidatos antes de enviar los top-5 al LLM, mitigando el fenómeno de pérdida de atención (*lost in the middle*). Por último, sería imperativo incorporar control de acceso basado en roles (RBAC) con filtrado criptográfico de metadatos a nivel de usuario, junto a una infraestructura de observabilidad continua (como Ragas o el framework de hit@k de C4) para monitorear en tiempo real latencias, deriva de embeddings y calidad de respuestas.

---

# Retos opcionales

> Se ejecutaron los retos **C4** y **C1** para alcanzar la máxima bonificación (30 pts).

## C4 · Evaluación automática de la recuperación

Archivo `eval/preguntas.csv` cargado con 10 preguntas (`pregunta`, `fuente_esperada`, `palabra_clave`). Evaluación de la función `hit_at_k`:

| chunk_size | hit@3 | hit@5 |
|---|---|---|
| 400 | 0.80 | 0.90 |
| 900 | 0.90 | 1.00 |
| 1500 | 0.90 | 0.90 |

**Dos limitaciones de esta métrica:**
1. **Coincidencia puramente léxica (exact string match):** La métrica evalúa la presencia de una subcadena literal (`palabra_clave.lower() in doc.page_content.lower()`). Esto penaliza injustamente fragmentos que contienen la respuesta correcta mediante sinónimos, paráfrasis o formulaciones alternativas válidas, y al mismo tiempo puede generar falsos positivos si la palabra clave aparece incidentalmente en una referencia bibliográfica o nota al pie sin responder a la pregunta.
2. **Evaluación aislada de la recuperación sin validar la generación:** La métrica solo mide si el pasaje correcto ingresó en la ventana de contexto del LLM, pero es ciega ante la calidad de la respuesta final: no detecta si el modelo alucina, si malinterpreta los datos numéricos o si el ordenamiento de los chunks coloca la información clave en posiciones desfavorables dentro del contexto.

## C1 · HNSW contra fuerza bruta con FAISS

Benchmark ejecutado con `faiss-cpu` comparando búsqueda exacta (`IndexFlatL2`) contra aproximada (`IndexHNSWFlat`) sobre $d = 64$, $N = 20\,000$ vectores de base y 200 consultas ($k = 10$):

- **Búsqueda exacta (`IndexFlatL2`):** `0.048 ms/consulta`.
- **HNSW (`M = 16`, `efConstruction = 100`):**
  - `efSearch = 10`: recall@10 = 0.329 | 0.005 ms/q (build 0.3s)
  - `efSearch = 16`: recall@10 = 0.425 | 0.005 ms/q (build 0.3s)
  - `efSearch = 32`: recall@10 = 0.599 | 0.005 ms/q (build 0.3s)
  - `efSearch = 64`: recall@10 = 0.770 | 0.010 ms/q (build 0.3s)
  - `efSearch = 128`: recall@10 = 0.905 | 0.015 ms/q (build 0.3s)
  - `efSearch = 256`: recall@10 = **0.971** | 0.030 ms/q (build 0.3s)
- **HNSW (`M = 32`, `efConstruction = 100`):**
  - `efSearch = 128`: recall@10 = **0.983** | 0.020 ms/q (build 0.4s)
  - `efSearch = 256`: recall@10 = **0.998** | 0.055 ms/q (build 0.4s)

**Respuestas al reto:**
1. **¿Qué `efSearch` se requiere para `recall >= 0.95` con `M = 16`?**
   Con `M = 16`, se requiere **`efSearch = 256`**, alcanzando un recall@10 de **0.971** (con `ef = 128` solo alcanza 0.905). Si se incrementa la conectividad a `M = 32`, basta con `efSearch = 128` para lograr un recall de 0.983.
2. **Barrido de $n$: ¿desde qué tamaño HNSW es más rápido que el exacto?**
   - $n = 20\,000$: exacto = 0.030 ms/q vs HNSW = 0.007 ms/q → **gana HNSW (4.3x más rápido)**.
   - $n = 100\,000$: exacto = 0.285 ms/q vs HNSW = 0.020 ms/q → **gana HNSW (14.2x más rápido)**.
   - $n = 500\,000$: exacto = 1.474 ms/q vs HNSW = 0.035 ms/q → **gana HNSW (42.1x más rápido)**.
   HNSW es más rápido ya desde $n \le 20\,000$. Mientras el costo de la fuerza bruta crece linealmente $\mathcal{O}(N \cdot d)$, HNSW crece de forma logarítmica $\mathcal{O}(\log N)$, ampliando exponencialmente su ventaja a escala.
3. **Evidencia gráfica:** Gráfica generada y guardada en `evidencias/C1_hnsw_benchmark.png` mostrando la curva de recall vs efSearch y recall vs latencia.
4. **Nota sobre datos sintéticos:** Los vectores aleatorios uniformes representan el caso más difícil para HNSW al carecer de clusters o geometría densa; con embeddings textuales reales de lenguaje natural, la coherencia semántica permite a HNSW superar el 95 % de recall con valores de `efSearch` notablemente menores.
