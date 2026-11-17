# Explorador Interactivo de Conocimiento

Aplicación web interactiva para estudiantes: procesa artículos de Wikipedia (o documentos propios), genera un grafo de relaciones estilo Obsidian y responde preguntas en lenguaje natural usando un LLM local (Ollama).

## Características

- **Extracción de Wikipedia**: obtiene el artículo principal y artículos relacionados vía MediaWiki API (`morelike:`)
- **Limpieza y chunking**: normaliza HTML, elimina referencias/citas, divide en chunks
- **Embeddings + NER**: vectores spaCy y entidades con `en_core_web_sm`
- **Recuperación TF-IDF**: encuentra chunks relevantes a cada pregunta
- **Respuestas con Ollama**: modelo local `qwen3:4b-instruct-2507-q4_K_M` — responde con el corpus procesado cuando hay contexto relevante, o con conocimiento general del modelo cuando no
- **Grafo interactivo**: nodos de artículos + conceptos compartidos detectados automáticamente (spaCy noun chunks), con física tipo Obsidian
- **Highlights selectivos**: al responder una pregunta, solo se iluminan los nodos realmente relevantes (máx. 4, por similitud TF-IDF + coincidencia de título) y la cámara hace zoom al área; las aristas no se modifican
- **UI estilo Zed/One Dark**: IBM Plex Sans, panel de preguntas + grafo, modal con chunks fuente

## Estructura del Proyecto

```
Proyecto NLP/
├── app.html                  # Interfaz web (panel preguntas + grafo)
├── server.py                 # Servidor HTTP + API (/api/ask, /api/process_topic, /api/upload, /api/process_uploads, /api/reset)
├── pipeline.py               # Pipeline: extracción → transformación → embeddings/NER
├── process_uploads.py        # Fusiona data/uploads/ con el corpus existente
├── generate_graph_data.py    # Genera graph_data.json (artículos + conceptos compartidos)
├── main.py                   # Menú CLI del pipeline
├── requirements.txt
├── data/
│   ├── raw/                  # JSONs crudos de Wikipedia
│   ├── processed/            # chunks, embeddings, entidades, graph_data.json
│   └── uploads/              # Documentos subidos por el usuario (.txt/.md/.pdf)
├── src/
│   ├── data_extraction/      # wikipedia_extractor.py (MediaWiki API) + document_loader.py (uploads por tema)
│   ├── transformation/       # data_transformation.py (limpieza, chunking, NLTK)
│   ├── embeddings/           # semantic_analyzer.py (spaCy vectors + NER)
│   └── generation/           # ollama_generator.py (TF-IDF retrieval + Ollama)
├── static/vendor/            # vis-network (grafo)
└── tests/                    # Tests unitarios (pytest)
```

## Instalación

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet punkt_tab
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve
```

## Uso

### Aplicación web (recomendado)

```bash
python server.py
```

Abrir `http://localhost:8000/app.html`, ingresar un tema, hacer clic en "Iniciar Exploración" y luego preguntar.

También se pueden subir documentos propios (`.txt`, `.md`, `.pdf`):

- Se guardan en `data/uploads/` (los nombres duplicados se renombran con `_1`, `_2`...)
- Se procesan **igual que los artículos de Wikipedia**: extracción de texto → limpieza → chunking → embeddings → NER → corpus y grafo
- **División por temas**: cada documento se divide automáticamente en sub-artículos según los temas detectados (frases nominales recurrentes vía spaCy). Un archivo puede generar varios nodos en el grafo con títulos `archivo · Tema`. Si no hay temas claros, queda como un solo nodo.
- **Con tema + archivos**: el pipeline incluye los uploads automáticamente en la misma corrida
- **Solo archivos** (sin tema): se fusionan con el corpus existente vía `process_uploads.py`
- Los documentos persisten en `data/uploads/` mientras el servidor esté corriendo y se incluyen en cada procesamiento de tema posterior
- **Ojo**: al iniciar `server.py` se eliminan todos los datos (`data/uploads/`, `data/raw/`, `data/processed/`) — cada sesión empieza limpia. También existe el botón "Nueva Exploración" en la interfaz que hace lo mismo sin reiniciar
- En el grafo, los nodos de documentos subidos se distinguen en **morado** y siempre son visibles (no se filtran por relevancia)

### Pipeline manual

```bash
python pipeline.py          # o: python main.py
python generate_graph_data.py
```

### Tests

```bash
pytest tests/ -v
```

## API

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/ask?q=...` | GET | Responder pregunta |
| `/api/ask` | POST | `{query: "..."}` |
| `/api/process_topic?topic=...` | GET | Ejecutar pipeline para un tema |
| `/api/upload` | POST | Guardar archivos en `data/uploads/` (multipart) |
| `/api/process_uploads` | POST | Fusionar uploads con el corpus y regenerar grafo |
| `/api/reset` | POST | Eliminar uploads + datos procesados (nueva exploración) |

## Notas

- Los datos en `data/` se regeneran con el pipeline; `data/raw/` y `data/processed/` están en `.gitignore`. `data/uploads/` también está ignorado (son archivos locales; usa `git add -f` si quieres versionar alguno).
- Ollama debe estar corriendo en `localhost:11434` para las preguntas.
- El modelo responde en el idioma de la pregunta.
- El generador de respuestas se recarga automáticamente cuando el corpus cambia (nuevo tema u upload).
