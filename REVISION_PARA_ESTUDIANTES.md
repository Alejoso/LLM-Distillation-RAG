# Revision General del Proyecto: LLM Distillation + RAG
**Fecha de revision:** Abril 2026  
**Dirigido a:** Estudiantes del equipo de investigacion

---

## 1. Estado Actual del Proyecto

El proyecto implementa un pipeline end-to-end para destilar un LLM grande (Llama-2-7B) en uno pequeno (TinyLlama-1.1B), utilizando RAG sobre documentos legales colombianos como fuente de contexto. El trabajo esta distribuido en 5 modulos funcionales, cada uno previamente desarrollado en ramas independientes.

### Modulos existentes y su nivel de madurez

| Modulo | Estado | Tests | Documentacion |
|--------|--------|-------|---------------|
| 01 - Data Collection (scraping + preprocesamiento) | Funcional | Si (9+ tests unitarios) | Buena |
| 02 - Data Processing (metricas + generacion de datos) | Funcional | Si (24+ tests) | Aceptable |
| 03 - RAG (vector DB + query + reranker) | Funcional | Casos de prueba manuales | Aceptable |
| 04 - Distillation Pipeline | Funcional | No tiene tests | Minima |
| 05 - LLM-as-Judge Evaluation | Funcional | No tiene tests | Buena (modelos Pydantic) |

---

## 2. Que se esta haciendo bien

### a) Arquitectura del pipeline
El flujo general tiene sentido y sigue una logica coherente: recolectar datos -> limpiar -> indexar -> generar datos de entrenamiento -> destilar -> evaluar. Cada etapa esta modularizada en su propio directorio, lo que facilita el desarrollo independiente.

### b) Calidad en la recoleccion de datos
- El web scraper tiene retry con exponential backoff y manejo de errores.
- El preprocesamiento de HTML es robusto: 40+ reglas regex para normalizar texto legal.
- El sistema de calidad (compute_metrics) tiene una rubrica cuantitativa clara (0-100 puntos) con tres dimensiones: ratio de lineas cortas, fragmentacion, e integridad de encabezados.
- Los archivos con score < 70 se separan automaticamente como "no usables". Esto es una buena practica.

### c) Tests en los modulos de datos
Los modulos 01 y 02 tienen tests unitarios bien escritos con la convencion Arrange-Act-Assert, cobertura de edge cases, y uso de pytest parametrize. Esto es un ejemplo a seguir para el resto del proyecto.

### d) Decision tecnica de destilacion
La eleccion de destilacion basada en logits con funcion de perdida hibrida (KL divergence + Cross-Entropy) esta bien fundamentada en el spike HU-RF-3 y alineada con la literatura (Hinton et al. 2015). El parametro alpha=0.7 prioriza los soft targets, lo cual es correcto cuando el teacher es significativamente mas capaz.

### e) RAG con two-stage retrieval
Usar BGE-M3 para embeddings + BGE-reranker-v2-m3 para reranking es una decision solida. El two-stage approach (retrieve many, rerank few) es el estado del arte para RAG en produccion.

### f) Evaluacion multidimensional
El LLM-as-Judge evalua en 4 dimensiones ponderadas (accuracy 40%, relevance 30%, completeness 20%, clarity 10%). Los pesos reflejan correctamente que la precision factual es lo mas importante para documentos legales.

---

## 3. Que se puede mejorar

### a) CRITICO: Tests en modulos 03, 04 y 05
Los modulos de RAG, distillation y evaluation **no tienen tests unitarios**. Esto es un riesgo significativo:

- **03_rag**: Hay `test_cases.json` con 20 casos de prueba, pero son para evaluacion manual, no tests automatizados. Falta: tests para el chunking, la extraccion de metadata, y el reranking.
- **04_distillation**: Es el modulo mas complejo y no tiene un solo test. Prioridad alta: testear `trainer.py` (la logica de loss), `dataset.py` (tokenizacion y padding), y `distill_data.py` (generacion de datos).
- **05_evaluation**: A pesar de usar modelos Pydantic (buena practica), no hay tests para el scoring ponderado ni para el parsing de respuestas del LLM.

**Recomendacion:** Cada modulo deberia tener al menos tests para las funciones puras (que no requieren GPU ni APIs externas). Usen mocks para las llamadas a modelos.

### b) CRITICO: Reproducibilidad
El pipeline actual tiene varios problemas de reproducibilidad:

- **No hay script unificado** que ejecute todo el pipeline de punta a punta. `run_pipeline.py` solo cubre la fase de destilacion, no las fases previas.
- **No hay Docker/containerizacion**: Las dependencias incluyen CUDA, modelos de HuggingFace, Ollama, ChromaDB, y spaCy. Sin un contenedor, reproducir el ambiente es fragil.
- **Los datos intermedios no estan versionados**: No hay DVC ni sistema similar para trackear datasets y modelos.
- **Semillas aleatorias**: `set_seed(42)` esta en el distillation pipeline, pero no en la generacion de datos ni en la creacion del vector DB.

**Recomendacion:** Crear un `Makefile` o script bash que ejecute el pipeline completo. Considerar DVC para versionado de datos. Agregar un `Dockerfile`.

### c) IMPORTANTE: El query.py de RAG usa gpt-5-nano como LLM para responder
En `03_rag/query_db/query.py`, el LLM que genera respuestas es `gpt-5-nano` (via OpenAI API), mientras que el pipeline de destilacion usa Llama-2-7B como teacher. Esto crea una inconsistencia: el modulo RAG standalone usa un modelo diferente al pipeline de destilacion.

**Recomendacion:** Unificar. El modulo RAG deberia ser configurable y poder usar el mismo teacher model que el pipeline de destilacion, o al menos documentar claramente por que se usan modelos diferentes en cada contexto.

### d) IMPORTANTE: Generacion de datos de entrenamiento
El script `generate_data_with_llm.py` genera 10 Q&A pairs por documento usando 9 tipos de preguntas. Observaciones:

- **Temperature=0.2, top_p=0.3**: Estos parametros son muy conservadores. Para generacion de datos de entrenamiento, podria ser util probar con temperatura ligeramente mas alta (0.4-0.6) para mayor diversidad, especialmente en preguntas de tipo "reescritura_simplificada".
- **Validacion minima**: Solo se valida longitud minima de instruccion (10 chars) y respuesta (30 chars). Falta validacion semantica: las respuestas generadas deberian validarse contra el documento fuente.
- **Sin deduplicacion**: No hay mecanismo para detectar si el LLM genera preguntas duplicadas o muy similares entre documentos.

**Recomendacion:** Agregar una fase de deduplicacion basada en embeddings (pueden reusar el BGE-M3 que ya tienen). Considerar validacion cruzada con el LLM-as-Judge.

### e) IMPORTANTE: La destilacion no usa gradient accumulation
En `trainer.py`, el batch_size=2 es muy pequeno para modelos de lenguaje. Aunque esto es probablemente por limitaciones de memoria GPU, no se implementa gradient accumulation, lo que significa que el effective batch size es 2.

**Recomendacion:** Implementar gradient accumulation para tener un effective batch size de al menos 16-32. Esto es sencillo: acumular gradientes durante N pasos antes de llamar `optimizer.step()`.

### f) MENOR: Estructura del proyecto
- Los `__pycache__` y `.pyc` fueron commiteados en la rama Distillation. Esto ya esta resuelto con el `.gitignore` unificado.
- El notebook `DistillationPipeLine.ipynb` no fue incluido en la integracion porque es una version anterior del pipeline modular. Si se necesita para presentaciones, deberia regenerarse desde el codigo modular.
- Las rutas en `config.py` son relativas a `BASE_DIR = Path(__file__).resolve().parent`, lo que funciona bien. Pero si se ejecutan scripts desde otros directorios, podrían fallar. Considerar usar variables de entorno.

### g) MENOR: Evaluacion del RAG
Las metricas de evaluacion del RAG (contextual relevancy, recall, precision via deepeval) son correctas, pero:
- Se evaluan **solo sobre una query** cada vez (no hay batch evaluation).
- No hay un script que ejecute todos los 20 test cases automaticamente y genere un reporte.

**Recomendacion:** Crear un script `evaluate_rag.py` que itere sobre `test_cases.json`, ejecute cada query, y genere metricas agregadas.

---

## 4. Tiene sentido lo que se esta haciendo?

**Si, el proyecto tiene sentido y esta bien encaminado.** Aqui esta el analisis:

### Por que tiene sentido
1. **Problema real**: Los documentos legales colombianos son un dominio especializado donde un LLM generalista grande es costoso de ejecutar. Destilar un modelo especializado mas pequeno tiene valor practico: menor latencia, menor costo de inferencia, posibilidad de ejecucion on-premise.

2. **RAG + Destilacion es una combinacion valida**: Usar RAG durante el entrenamiento (para que el teacher genere respuestas contextualizadas) y potencialmente durante la inferencia del student, permite que el modelo pequeno se beneficie de informacion actualizada sin necesidad de re-entrenar.

3. **El enfoque de logits-based distillation esta justificado**: Para un dominio donde la precision factual es critica (documentos legales), transferir la distribucion completa de probabilidades del teacher (no solo el texto generado) le da al student mas informacion para aprender.

4. **LLM-as-Judge es apropiado para este dominio**: Evaluar manualmente miles de respuestas sobre derecho colombiano requeriria expertos legales. Un LLM como juez automatizado, con las dimensiones correctas (accuracy como prioridad), es una solucion practica.

### Riesgos a considerar
1. **El student (TinyLlama-1.1B) puede ser demasiado pequeno** para el dominio legal. Si los resultados no son satisfactorios, considerar modelos intermedios como Phi-2 (2.7B) o Mistral-7B con quantizacion (que seria mas rapido pero con el mismo numero de parametros).

2. **El teacher (Llama-2-7B) no es el mejor modelo disponible**. Llama-3 o Mistral-7B-v0.3 son significativamente mejores. Si el teacher genera respuestas de baja calidad, el student heredara esos errores. Considerar actualizar el teacher.

3. **No hay baseline de comparacion**. Para validar que la destilacion funciona, necesitan comparar:
   - Student destilado vs Student sin destilar (fine-tuned directamente)
   - Student destilado vs Teacher original
   - Student con RAG vs Student sin RAG
   Sin estos baselines, no pueden cuantificar el aporte de cada componente.

4. **Legal compliance**: Usando modelos de Meta (Llama-2) para procesar documentos legales colombianos, asegurense de cumplir con las licencias de los modelos y las regulaciones de datos aplicables.

---

## 5. Hoja de ruta sugerida (proximos pasos)

### Prioridad Alta
- [ ] Agregar tests unitarios a modulos 03, 04, 05
- [ ] Implementar gradient accumulation en el trainer
- [ ] Crear baselines de comparacion (student sin destilar, teacher directo)
- [ ] Script de evaluacion automatica del RAG sobre todos los test cases
- [ ] Evaluar si TinyLlama-1.1B es suficiente o si se necesita un student mas grande

### Prioridad Media
- [ ] Crear Dockerfile para reproducibilidad
- [ ] Unificar el LLM usado en RAG standalone vs pipeline de destilacion
- [ ] Agregar deduplicacion en la generacion de datos de entrenamiento
- [ ] Implementar evaluacion end-to-end automatizada (generate -> evaluate -> report)
- [ ] Considerar actualizar el teacher a Llama-3 o Mistral-7B

### Prioridad Baja
- [ ] Agregar DVC para versionado de datos y modelos
- [ ] Regenerar notebook desde codigo modular (para presentaciones)
- [ ] Implementar logging estructurado con wandb o mlflow
- [ ] Explorar quantizacion (GPTQ/AWQ) como alternativa a destilacion

---

## 6. Resumen ejecutivo

El proyecto tiene una arquitectura solida y un flujo logico bien definido. Los modulos de recoleccion y procesamiento de datos son los mas maduros, con buena cobertura de tests. Los modulos de destilacion y evaluacion son funcionales pero necesitan tests y mejoras en reproducibilidad. La decision tecnica mas importante es validar si TinyLlama-1.1B es un student adecuado para el dominio legal, lo cual requiere los baselines de comparacion mencionados. El trabajo realizado hasta ahora es una base solida sobre la cual construir un articulo de investigacion.
