# SI3003 — Introducción a la Inteligencia Artificial

Entregables del curso. Autor: **Juan Camilo Gómez** (`jcgomezj`).

> Declaración de uso de IA generativa: se utilizó un asistente de IA como apoyo en la implementación y redacción; la responsabilidad técnica y conceptual del contenido es del autor.

## Talleres por semana

| Semana | Tema | Carpeta | Entregable |
|---|---|---|---|
| 4 | MDPs | `Semana4-MDPs/` | Laboratorio del robot de almacén: modelado del MDP + Value/Policy Iteration, interpretación y experimentos. |
| 5 | Aprendizaje por refuerzo | `Semana5-QLearning/` | Q-Learning en Taxi-v4: `choose_action`, `update_q`, `train_q_learning`, curva de aprendizaje, visualización y 7 preguntas. |
| 6 | Machine Learning | `Semana6-MachineLearning/` | Challenge de clasificación de vinos en 3 etapas (datos/EDA, entrenamiento y selección, evaluación e inferencia). |
| 7 | Redes neuronales (Keras 3) | `Semana7-RedesNeuronales/` | MLP sobre CIFAR-10 en escala de grises y sobre Olivetti Faces. |
| 8 | CNN / Transfer Learning | `Semana8-CNN-TransferLearning/` | Taller de transfer learning con dataset propio (EfficientNetB0) + fine-tuning y análisis de errores. |

## Notas

- Los notebooks están ejecutados con sus salidas.
- **Semana 6:** incluye `data/`, `artifacts/` y `reports/` con los artefactos del pipeline (modelo campeón, contrato de datos, métricas).
- **Semana 8:** las carpetas de imágenes (`data_raw/`, `dataset/`) **no** se incluyen por tamaño; se regeneran al ejecutar el notebook (descarga las imágenes con `ddgs`).

## Dependencias

```bash
pip install numpy "gymnasium[toy-text]" scikit-learn seaborn joblib tensorflow-cpu keras ddgs matplotlib pillow
```
