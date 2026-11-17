# Explorador Interactivo de Conocimiento

Aplicación web interactiva para estudiantes: procesa artículos de Wikipedia (o documentos propios), genera un grafo de relaciones estilo Obsidian y responde preguntas en lenguaje natural usando un LLM local (Ollama).

## Características

- **Extracción de Wikipedia**: obtiene el artículo principal y artículos relacionados vía MediaWiki API (`morelike:`)
- **Limpieza y chunking**: normaliza HTML, elimina referencias/citas, divide en chunks
- **Embeddings + NER**: vectores spaCy y entidades con `en_core_web_sm`
- **Recuperación TF-IDF**: encuentra chunks relevantes a cada pregunta
- **Respuestas con Ollama**: modelo local `qwen3:4b-instruct-2507-q4_K_M` — responde con el corpus procesado cuando hay contexto relevante, o con conocimiento general del modelo cuando no
- **Grafo interactivo**: nodos de artículos + conceptos compartidos detectados automáticamente (spaCy noun chunks), con física tipo Obsidian
- **UI estilo Zed/One Dark**: IBM Plex Sans, panel de preguntas + grafo, modal con chunks fuente

## Estructura del Proyecto

```
Proyecto NLP/
├── app.html                  # Interfaz web (panel preguntas + grafo)
├── server.py                 # Servidor HTTP + API (/api/ask, /api/process_topic, /api/upload)
├── pipeline.py               # Pipeline: extracción → transformación → embeddings/NER
├── generate_graph_data.py    # Genera graph_data.json (artículos + conceptos compartidos)
├── main.py                   # Menú CLI del pipeline
├── requirements.txt
├── data/
│   ├── raw/                  # JSONs crudos de Wikipedia
│   └── processed/            # chunks, embeddings, entidades, graph_data.json
├── src/
│   ├── data_extraction/      # wikipedia_extractor.py (MediaWiki API)
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
- **Con tema + archivos**: el pipeline incluye los uploads automáticamente en la misma corrida
- **Solo archivos** (sin tema): se fusionan con el corpus existente vía `process_uploads.py`
- Los documentos persisten en `data/uploads/` y se incluyen en cada procesamiento de tema posterior

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

## Notas

- Los datos en `data/` se regeneran con el pipeline; están en `.gitignore`.
- Ollama debe estar corriendo en `localhost:11434` para las preguntas.
- El modelo responde en el idioma de la pregunta.
