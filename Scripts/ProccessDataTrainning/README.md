# GeneratingDataWithLLM — Documentación

Scripts para generar un dataset de entrenamiento a partir de documentos legales colombianos usando un LLM.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `GeneratingDataWithLLM.py` | Script principal para correr en Apolo (HuggingFace + GPU) |
| `GeneratingDataWithLLM_ollama_test.py` | Versión de prueba local con Ollama |
| `GeneratingDataWithLLM.ipynb` | Notebook original usado en Google Colab/Cloud |

---

## GeneratingDataWithLLM.py (Apolo)

### Flags

| Flag | Requerido | Default | Descripción |
|---|---|---|---|
| `--laws_folder` | Sí | — | Ruta a la carpeta con los `.txt` de leyes limpias |
| `--output_file` | Sí | — | Ruta del archivo JSON de salida |
| `--hf_token` | Sí | — | Token de HuggingFace (necesario para el modelo Llama-2 que es gated) |
| `--max_files` | No | todos | Número máximo de archivos a procesar |

### Uso

```bash
python GeneratingDataWithLLM.py \
    --laws_folder /ruta/dataCleaned/Laws \
    --output_file /ruta/dataCleaned/datasetTrain.json \
    --hf_token hf_xxxxxxxxxxxx \
    --max_files 100
```

---

## GeneratingDataWithLLM_ollama_test.py (local)

### Flags

| Flag | Requerido | Default | Descripción |
|---|---|---|---|
| `--laws_folder` | No | `../../dataCleaned/Laws` | Ruta a la carpeta con los `.txt` de leyes limpias |
| `--output_file` | No | `../../dataCleaned/datasetTrain_test.json` | Ruta del archivo JSON de salida |
| `--model` | No | `llama3` | Nombre del modelo Ollama a usar |
| `--max_files` | No | `1` | Número máximo de archivos a procesar |

### Uso

```bash
# Con defaults (1 archivo, llama3, rutas automáticas)
python GeneratingDataWithLLM_ollama_test.py

# Especificando todo
python GeneratingDataWithLLM_ollama_test.py \
    --laws_folder dataCleaned/Laws \
    --output_file dataCleaned/datasetTrain_test.json \
    --model llama3 \
    --max_files 3
```

> Requiere Ollama corriendo: `ollama serve`

---

## Cómo funciona

```
.txt legal  →  [generar prompt]  →  LLM  →  [parsear output]  →  [validar]  →  .json
```

### 1. Lectura del documento
- Lee archivos `.txt` de `laws_folder`
- Si el archivo tiene la sección `CONTENIDO:`, extrae solo esa parte
- Trunca a 4000 caracteres para no exceder el contexto del modelo

### 2. Generación con el LLM
El prompt instruye al modelo a generar **exactamente 10 ejemplos** con esta distribución de tipos:

| Tipo | Ejemplos |
|---|---|
| `idea_central` | 2 |
| `resumen_3_niveles` | 1 |
| `esencial_vs_accesorio` | 1 |
| `estructura_logica` | 1 |
| `reescritura_simplificada` | 1 |
| `intencion_autor` | 1 |
| `conceptos_clave` | 1 |
| `reduccion_extrema` | 1 |
| `conexiones_internas` | 1 |

Parámetros de generación: `temperature=0.2`, `top_p=0.3`, `max_new_tokens=3000`

### 3. Parseo
El output del modelo se divide por `Ejemplo N:` y de cada bloque se extrae:
- `Instruction:` — la pregunta o tarea
- `Response:` — la respuesta sintetizada
- `context` — siempre vacío (por diseño del dataset)

### 4. Validación
Un ejemplo se descarta si:
- La instrucción tiene menos de 10 caracteres
- La respuesta tiene menos de 30 caracteres
- El campo `context` no está vacío

### 5. Guardado
Solo se guarda un archivo si tiene **exactamente 10 ejemplos válidos**. El resultado se acumula en un JSON con la estructura:

```json
{
  "data": [
    {
      "instruction": "...",
      "context": "",
      "response": "..."
    }
  ]
}
```

---

## Modelos

| Script | Modelo |
|---|---|
| `GeneratingDataWithLLM.py` | `meta-llama/Llama-2-7b-chat-hf` (HuggingFace, requiere GPU) |
| `GeneratingDataWithLLM_ollama_test.py` | `llama3` (Ollama local, corre en CPU) |
