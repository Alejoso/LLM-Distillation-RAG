# Destilacion de Modelos de Lenguaje con Generacion Aumentada por Recuperacion para el Dominio Legal Colombiano

**Equipo de Investigacion**  
Universidad - Departamento de Ingenieria de Sistemas, Colombia

---

## Resumen

Los modelos de lenguaje grandes (LLMs) han demostrado capacidades notables en tareas de comprension y generacion de texto en dominios especializados. Sin embargo, su alto costo computacional de inferencia limita su despliegue en escenarios con recursos restringidos. Este trabajo presenta un pipeline integral para la destilacion de conocimiento de un modelo Llama-2-7B hacia un modelo TinyLlama-1.1B, aplicado al dominio de documentos legales colombianos. Nuestro enfoque integra Generacion Aumentada por Recuperacion (RAG) durante la fase de entrenamiento del profesor, un sistema de control de calidad para los datos de entrenamiento, y evaluacion automatica mediante LLM-as-Judge. Describimos la arquitectura completa del pipeline --desde la recoleccion y procesamiento de documentos del portal SUIN hasta la evaluacion del modelo destilado-- y discutimos las decisiones tecnicas, resultados preliminares y direcciones futuras de investigacion.

**Palabras clave:** Destilacion de conocimiento, modelos de lenguaje grandes, RAG, documentos legales, NLP en espanol

---

## 1. Introduccion

El procesamiento de lenguaje natural aplicado al dominio legal ha ganado relevancia debido a la complejidad y volumen de los marcos normativos. En Colombia, el Sistema Unico de Informacion Normativa (SUIN) alberga miles de documentos legales cuyo analisis automatizado podria beneficiar a profesionales del derecho, entidades gubernamentales y ciudadanos.

Los LLMs como GPT-4, Llama-2 y Mistral han demostrado capacidades sobresalientes en comprension legal [1]. Sin embargo, desplegar modelos de 7B+ parametros en produccion implica costos significativos de infraestructura y latencia elevada, particularmente en contextos donde el procesamiento on-premise es un requisito.

La **destilacion de conocimiento** [2] ofrece una solucion: transferir el conocimiento de un modelo grande (*teacher*) a uno mas pequeno (*student*), preservando la mayor parte de las capacidades con una fraccion de los parametros. Este trabajo extiende el paradigma de destilacion con dos innovaciones contextuales:

1. **RAG-augmented distillation**: El modelo teacher genera sus respuestas con acceso a un sistema RAG sobre la base legal colombiana, produciendo datos de entrenamiento mas precisos y contextualizados.
2. **Pipeline integral con control de calidad**: Desde la recoleccion de datos hasta la evaluacion, cada etapa incluye mecanismos de validacion y metricas cuantitativas.

---

## 2. Trabajo Relacionado

### 2.1 Destilacion de Conocimiento en LLMs

Hinton et al. [2] propusieron la destilacion de conocimiento usando *soft targets* con un parametro de temperatura. Desde entonces, se han desarrollado multiples variantes:

- **Basada en logits**: El student aprende a aproximar la distribucion de probabilidades del teacher [2, 3]. Es la tecnica mas directa y la que mejor preserva la incertidumbre del teacher.
- **Basada en features**: El student replica representaciones intermedias del teacher [4, 5]. Requiere arquitecturas compatibles.
- **Basada en respuestas**: El student se entrena directamente sobre texto generado por el teacher [6]. Mas simple pero pierde informacion distribucional.

DistilBERT [7] demostro que un modelo 40% mas pequeno puede retener 97% del rendimiento del teacher en tareas de comprension. TinyBERT [5] extendio esto con destilacion en multiples capas.

### 2.2 RAG (Retrieval-Augmented Generation)

Lewis et al. [8] introdujeron RAG como un mecanismo para fundamentar las respuestas de LLMs en fuentes externas, reduciendo alucinaciones. En el dominio legal, RAG es particularmente valioso porque las respuestas deben estar ancladas en normas especificas.

Trabajos recientes han explorado el uso de embeddings multilingues como BGE-M3 [9] y tecnicas de reranking de dos etapas [10] para mejorar la precision de la recuperacion.

### 2.3 LLM como Juez

Zheng et al. [11] propusieron usar LLMs como jueces automaticos para evaluar calidad de respuestas, demostrando alta correlacion con evaluacion humana.

---

## 3. Metodologia

### 3.1 Arquitectura General del Pipeline

El pipeline se compone de cinco etapas secuenciales:

1. Recoleccion y preprocesamiento de documentos
2. Indexacion en base de datos vectorial (RAG)
3. Generacion de datos de entrenamiento
4. Destilacion de conocimiento
5. Evaluacion automatica

```
[SUIN Portal] --scrape--> [Raw HTML] --preprocess--> [Clean TXT + Quality Score]
                                                              |
                                                    +---------+---------+
                                                    |                   |
                                               index in DB        generate Q&A
                                                    |            (via teacher LLM)
                                                    v                   |
                                              [Chroma Vector DB]       |
                                                    |                   |
                                                    +-----+-------------+
                                                          |
                                                    RAG-augmented
                                                    teacher inference
                                                          |
                                                          v
                                                  [Distillation Data]
                                                   (prompts + logits)
                                                          |
                                                     train student
                                                   (KL div + CE loss)
                                                          |
                                                          v
                                                  [TinyLlama Fine-tuned]
                                                          |
                                                     evaluate with
                                                     LLM-as-Judge
                                                          |
                                                          v
                                                  [Quality Metrics]
```

### 3.2 Recoleccion y Preprocesamiento de Datos

Los documentos legales se obtienen del portal SUIN mediante web scraping con mecanismos de retry y backoff exponencial. Cada documento HTML pasa por un pipeline de limpieza que incluye:

- Eliminacion de elementos no textuales (scripts, estilos, navegacion)
- Normalizacion de texto legal (40+ reglas regex para manejo de articulos, valores monetarios, listas, y estructura legal)
- Extraccion de metadatos (tipo, numero, ano, entidad emisora, estado)

Se implementa un sistema de **scoring de calidad** con tres dimensiones ponderadas:

> Q = w_l * S_lines + w_f * S_frag + w_h * S_header

donde S_lines es el ratio de lineas cortas (peso 45%), S_frag la fragmentacion (45%), y S_header la integridad de encabezados (10%). Documentos con Q < 70 se descartan automaticamente.

### 3.3 Sistema RAG

Los documentos limpios se indexan en una base de datos vectorial ChromaDB usando el modelo de embeddings BGE-M3 [9], seleccionado por su rendimiento en tareas multilingues.

**Chunking:** Se utiliza `RecursiveCharacterTextSplitter` con tamano de chunk de 800 tokens y overlap de 150 tokens, preservando la coherencia de articulos legales.

**Recuperacion en dos etapas:**
1. Busqueda por similitud en ChromaDB (k=10)
2. Reranking con BGE-reranker-v2-m3 (k_final=2)

### 3.4 Generacion de Datos de Entrenamiento

Se utiliza el modelo teacher (Llama-2-7B-chat) para generar pares de instrucciones-respuestas a partir de los documentos legales. Se definen 9 tipos de tareas de comprension:

1. Idea central
2. Resumen en 3 niveles
3. Esencial vs. accesorio
4. Estructura logica
5. Reescritura simplificada
6. Intencion del autor
7. Conceptos clave
8. Reduccion extrema
9. Conexiones internas

Cada documento genera exactamente 10 pares, con validacion de longitud minima (instruccion >= 10 caracteres, respuesta >= 30 caracteres).

### 3.5 Destilacion de Conocimiento

Se emplea destilacion basada en logits con una funcion de perdida hibrida:

> L = alpha * L_soft + (1 - alpha) * L_hard

donde:

> L_soft = T^2 * KL_divergence(softmax(z_s / T) || softmax(z_t / T))

> L_hard = CrossEntropy(z_s, y)

con T=4.0 (temperatura), alpha=0.7, z_s y z_t los logits del student y teacher respectivamente.

**Almacenamiento eficiente de logits:** Dado que la matriz completa de logits del teacher tiene dimensiones L x V (longitud de secuencia x tamano de vocabulario), almacenarla directamente resulta prohibitivo. Para un vocabulario de 32,000 tokens y longitud de secuencia de 2,048, cada muestra generaria ~500 MB en formato texto. En su lugar, se almacenan unicamente los top-K logits (K=50) por posicion en formato binario (.pt, float16), reduciendo el almacenamiento por muestra de ~500 MB a ~800 KB — una reduccion de ~600x. Durante el entrenamiento, el tensor completo se reconstruye mediante `scatter_`, asignando un valor de piso (-10) a las posiciones no cubiertas por el top-K, lo cual preserva la forma de la distribucion softmax sin afectar significativamente la divergencia KL.

**Configuracion de entrenamiento:**

| Parametro | Valor |
|-----------|-------|
| Teacher | meta-llama/Llama-2-7b-chat-hf (~7B params) |
| Student | TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1B params) |
| Optimizer | AdamW, lr = 2e-5 |
| Batch size | 2 |
| Epocas | 3 |
| Max sequence length | 2048 tokens |
| Top-K logits almacenados | 50 |
| Alpha | 0.7 |
| Temperature | 4.0 |

**RAG-augmented distillation:** Opcionalmente, durante la generacion de datos de destilacion, el teacher recibe contexto recuperado del sistema RAG. Esto produce respuestas mas fundamentadas en la normativa, que el student aprende a emular.

#### Algoritmo: Pipeline de Destilacion con RAG

```
1. Cargar datos procesados D = {(x_i)}
2. Cargar teacher M_T y student M_S
3. Para cada instruccion x_i en D:
   a. Si RAG habilitado:
      - c_i = Retrieve(x_i)       # Contexto RAG
      - p_i = BuildPrompt(x_i, c_i)
   b. Si no:
      - p_i = BuildPrompt(x_i)
   c. y_i, z_t_i = M_T(p_i)       # Respuesta + logits del teacher
   d. z_hat_t_i = TopK(z_t_i, K)  # Retener top-K logits
   e. Guardar (p_i, y_i) en JSONL; z_hat_t_i en .pt binario
4. Para cada epoca = 1 to E:
   a. Para cada batch B del dataset:
      - z_t = Reconstruct(z_hat_t, V)  # Expandir a vocabulario completo
      - z_s = M_S(B)                   # Logits del student
      - L = alpha * L_soft + (1 - alpha) * L_hard
      - Actualizar M_S con gradientes de L
```

### 3.6 Evaluacion con LLM-as-Judge

El modelo destilado se evalua mediante un framework de LLM-as-Judge que puntua las respuestas en cuatro dimensiones:

| Dimension | Peso | Escala |
|-----------|------|--------|
| Accuracy (precision factual) | 40% | 1-5 |
| Relevance (pertinencia) | 30% | 1-5 |
| Completeness (cobertura) | 20% | 1-5 |
| Clarity (claridad) | 10% | 1-5 |

La puntuacion ponderada final: **S = 0.4 * S_a + 0.3 * S_r + 0.2 * S_c + 0.1 * S_cl**

Adicionalmente, se evalua la calidad del sistema RAG usando metricas estandar de recuperacion: Contextual Relevancy, Contextual Recall, y Contextual Precision.

---

## 4. Resultados Preliminares

### 4.1 Recoleccion de Datos

El pipeline de recoleccion proceso documentos del portal SUIN, clasificando cada archivo segun su puntaje de calidad. Los documentos con puntaje >= 70 se consideran usables para entrenamiento.

### 4.2 Datos de Entrenamiento

Se generaron pares de instruccion-respuesta cubriendo los 9 tipos de tareas de comprension definidos, proporcionando diversidad en el entrenamiento del modelo student.

*Nota: Los resultados cuantitativos completos de destilacion y evaluacion se reportaran en la version final del articulo una vez completados los experimentos con los baselines de comparacion.*

---

## 5. Discusion

### 5.1 Ventajas del Enfoque

- **Pipeline integral**: A diferencia de trabajos que se enfocan solo en la destilacion, nuestro enfoque cubre todo el ciclo de vida.
- **Control de calidad en datos**: El sistema de scoring automatico reduce el ruido en los datos de entrenamiento.
- **RAG durante destilacion**: El teacher produce respuestas mas fundamentadas, lo que potencialmente mejora la calidad del student.
- **Dominio especifico**: El enfoque en documentos legales colombianos demuestra aplicabilidad a idiomas y dominios poco representados en la literatura.

### 5.2 Limitaciones

- **Tamano del student**: TinyLlama-1.1B puede ser insuficiente para capturar la complejidad del lenguaje legal. Modelos intermedios (2-3B parametros) podrian ofrecer mejor balance.
- **Teacher no optimo**: Llama-2-7B no es el modelo mas capaz disponible. Modelos mas recientes (Llama-3, Mistral) como teachers podrian mejorar la calidad de la destilacion.
- **Evaluacion humana pendiente**: La evaluacion por LLM-as-Judge debe validarse contra juicio de expertos legales.
- **Effective batch size**: El batch size de 2 sin gradient accumulation puede afectar la convergencia.

### 5.3 Direcciones Futuras

1. Evaluar modelos student de mayor capacidad (Phi-2, Mistral-7B con cuantizacion)
2. Implementar destilacion progresiva (7B -> 3B -> 1B)
3. Explorar destilacion basada en features ademas de logits
4. Comparar con fine-tuning directo del student (sin destilacion)
5. Validar con evaluadores humanos expertos en derecho colombiano
6. Explorar cuantizacion (GPTQ, AWQ) como alternativa complementaria

---

## 6. Conclusion

Este trabajo presenta un pipeline completo para la destilacion de LLMs en el dominio legal colombiano, integrando RAG, control de calidad de datos, y evaluacion automatica. La arquitectura modular permite la iteracion independiente sobre cada componente, facilitando la experimentacion y la mejora continua. Los resultados preliminares demuestran la viabilidad del enfoque, y los experimentos futuros con baselines de comparacion permitiran cuantificar el aporte especifico de cada componente del pipeline.

---

## Agradecimientos

Los autores agradecen el acceso a recursos computacionales de HPC (Apolo) y al portal SUIN por la disponibilidad publica de documentos legales colombianos.

---

## Referencias

[1] D. M. Katz et al., "GPT-4 passes the bar exam," *Philosophical Transactions of the Royal Society A*, vol. 382, no. 2270, 2024.

[2] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv preprint arXiv:1503.02531*, 2015.

[3] Y. Kim and A. M. Rush, "Sequence-level knowledge distillation," in *EMNLP*, 2016, pp. 1317-1327.

[4] A. Romero et al., "FitNets: Hints for thin deep nets," in *ICLR*, 2015.

[5] X. Jiao et al., "TinyBERT: Distilling BERT for natural language understanding," in *Findings of EMNLP*, 2020, pp. 4163-4174.

[6] R. Taori et al., "Stanford Alpaca: An instruction-following LLaMA model," 2023.

[7] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv preprint arXiv:1910.01108*, 2019.

[8] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *NeurIPS*, vol. 33, 2020, pp. 9459-9474.

[9] J. Chen et al., "BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation," *arXiv preprint arXiv:2402.03216*, 2024.

[10] R. Nogueira and K. Cho, "Passage re-ranking with BERT," *arXiv preprint arXiv:1901.04085*, 2019.

[11] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," in *NeurIPS*, vol. 36, 2023.
