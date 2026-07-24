# Quiz 01: Introducción a la Inteligencia Artificial
# Estudiante: Juan Camilo Gómez Jiménez
**Materia:** Introducción a la Inteligencia Artificial  
**Universidad:** Universidad EAFIT  
**Sujeto de Análisis (Space):** Wan 2.2 I2V (14B) — Fast  

---

## 1. Descripción del Space
* **Nombre:** Wan 2.2 I2V (14B) — Fast
* **Enlace:** https://huggingface.co/spaces/Wan-AI/Wan2.1-I2V-14B-480P
* **¿Qué hace el agente?:** Es un sistema generativo enfocado en la tarea *Image-to-Video* (I2V) que toma una imagen fija de entrada (y un *prompt* de texto opcional) para sintetizar una secuencia de video temporalmente coherente utilizando una arquitectura de difusión latente (*Latent Video Diffusion Transformer*).

---

## 2. Análisis PEAS

| Elemento | Pregunta Guía | Aplicación en Wan 2.2 I2V (14B) — Fast |
| :--- | :--- | :--- |
| **Performance** | ¿Qué significa que el agente haga bien su trabajo? | Alta coherencia temporal del movimiento, alineación con la imagen original (*I2V alignment*), adherencia al *prompt*, alta resolución/FPS, baja latencia de inferencia y ausencia de artefactos visuales. |
| **Environment** | ¿Con qué interactúa el agente? | Espacio latente visual, entorno de ejecución en GPU (CUDA/PyTorch), memoria VRAM e interfaz/API de Hugging Face Spaces. |
| **Actuators** | ¿Qué acciones produce? | Pasos de reducción de ruido latente (*de-noising*), capas de atención temporal/espacial, decodificador VAE para sintetizar píxeles y exportación del archivo de video (MP4). |
| **Sensors** | ¿Qué información recibe como entrada? | Codificador visual (*Vision Encoder* para la imagen), codificador de texto (*Text Encoder*) y parámetros de configuración (*seed*, *CFG scale*, pasos de inferencia). |

---

## 3. Clasificación del Entorno de Trabajo

| Propiedad | Clasificación | Justificación |
| :--- | :--- | :--- |
| **Observable** | Parcialmente | Solo percibe los píxeles de la imagen 2D fija y el texto. No posee contexto tridimensional ni conoce la historia previa/posterior a la toma. |
| **Determinista** | No (Estocástico) | La generación parte de muestreos probabilísticos con ruido gaussiano. Cambiar la semilla (*seed*) genera animaciones completamente distintas para la misma imagen. |
| **Episódico** | No (Secuencial) | El proceso de reducción de ruido (*denoising*) se ejecuta paso a paso; el tensor en el paso $t$ depende del estado en $t-1$. |
| **Estático** | Sí | La imagen y los parámetros de entrada no cambian ni se modifican mientras la GPU procesa el video. |
| **Discreto** | No (Continuo) | Opera en espacios latentes continuos con tensores de punto flotante ($FP16/BF16$) y representaciones continuas de luz, color y movimiento. |
| **Conocido** | Sí | Las reglas del entorno (mecanismos de difusión latente, leyes de procesamiento de tensores y arquitectura del Transformer) están completamente definidas dentro del sistema. |

---

## 4. ¿Qué tipo de programa de agente creen que es?

Implementa un **Agente basado en modelos** y **Agente con aprendizaje**:
* **Basado en modelos:** Utiliza bloques de atención temporal que actúan como un modelo interno del mundo, prediciendo cómo deben evolucionar la física y la estructura de los objetos a lo largo del tiempo.
* **Con aprendizaje:** Sus pesos fueron entrenados previamente a partir de volúmenes masivos de datos de video para aprender distribuciones cinematográficas y cinemáticas.

---

## 5. Discusión en Clase

### ¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno?
**Sí.** Por ejemplo, un Space de *Wan 2.2 I2V* y uno de *Runway Gen-2* o *Sora* comparten el mismo tipo de entorno (**parcialmente observable, estocástico, secuencial, estático, continuo y conocido**), ya que ambos enfrentan la misma tarea conceptual: generar video mediante difusión a partir de entradas estáticas.

### ¿Es posible saber con certeza qué tipo de agente implementa un Space únicamente observándolo?
**No.** La observación externa solo revela el *comportamiento observable* (entrada $\rightarrow$ salida). Dos Spaces pueden ofrecer la misma interfaz (recibir imagen, entregar video), pero internamente uno puede ser un Transformer de difusión (*Agente basado en modelos*), otro una red GAN o una secuencia de filtros heurísticos (*Agente de reflejo simple*).

### ¿Qué diferencia existe entre el comportamiento observable de un agente y su implementación interna?
* **Comportamiento Observable:** Es la función de mapeo externa percibida por el usuario (relación entre los datos ingresados y el archivo resultante).
* **Implementación Interna:** Es el programa interno del agente (algoritmos, tensores, redes neuronales y representaciones de datos) que procesa las percepciones para generar las acciones.

---

## 6. Reto Adicional: Clasificación de Spaces

### Space 1: Totalmente observable, determinista y episódico
* **Ejemplo de Space:** Un Space de **Clasificación de Imágenes (ej. ResNet-50 / MNIST)** o un **Solucionador de 8-Puzzle en un solo paso**.
* **Justificación:**
  * **Totalmente observable:** El agente percibe la totalidad de la entrada (todos los píxeles del gráfico o el estado del tablero) de una sola vez.
  * **Determinista:** La misma entrada procesada por el mismo algoritmo/modelo siempre produce exactamente la misma predicción o solución.
  * **Episódico:** La tarea de clasificar una imagen es un evento aislado; no depende de imágenes procesadas previamente ni altera las consultas futuras.

### Space 2: Parcialmente observable, estocástico y secuencial
* **Ejemplo de Space:** **Wan 2.2 I2V (14B) — Fast**.
* **Justificación:**
  * **Parcialmente observable:** Solo recibe un encuadre 2D estático sin información 3D ni antecedentes temporales.
  * **Estocástico:** Produce salidas diferentes según la semilla (*seed*) y el ruido inicial.
  * **Secuencial:** Cada iteración del proceso de *denoising* determina directamente la calidad del tensor para el siguiente paso del proceso.