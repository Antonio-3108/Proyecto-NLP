# Proyecto Final — Explorador Interactivo de Conocimiento

## 1. Idea

### Producto

Se desarrolló un **Explorador Interactivo de Conocimiento** en formato de aplicación web orientado al estudio. La aplicación permite ingresar un tema (por ejemplo, "Halo videogame") o subir documentos propios, y a partir de ellos:

- extrae y procesa contenido de **Wikipedia** y/o de los **documentos del usuario**;
- construye una **base de conocimiento** con chunks, embeddings, entidades y conceptos;
- genera un **grafo de relaciones** interactivo (estilo Obsidian Graph View);
- responde **preguntas en lenguaje natural** (español e inglés) usando un LLM local (Ollama), fundamentado en los documentos procesados o en su conocimiento general.

El objetivo no es sustituir el material de estudio, sino **convertirlo en una forma más sencilla de consultar, comprender y explorar**.

### Usuario final

**Estudiantes** que necesitan comprender un tema nuevo o repasar material de estudio y desean encontrar información relevante sin tener que revisar manualmente grandes cantidades de contenido.

### Problema que resuelve

Los estudiantes pueden disponer de múltiples documentos, apuntes o materiales de clase, pero encontrar rápidamente una explicación concreta y entender qué conceptos están relacionados puede requerir revisar una gran cantidad de información.

El producto busca reducir este esfuerzo permitiendo que el estudiante consulte directamente el material mediante lenguaje natural, explorar visualmente las relaciones entre artículos y conceptos, y descubrir puntos de partida para continuar investigando.

---

## 2. Justificación

### ¿Por qué vale la pena?

Los estudiantes tienen acceso a una gran cantidad de información, pero encontrar lo relevante y relacionar los conceptos puede requerir revisar manualmente múltiples documentos. El producto busca **reducir ese esfuerzo y facilitar una exploración más rápida y accesible del material de estudio**.

Por ejemplo, un estudiante podría procesar el tema "Chandelure" y preguntar:

```text
> ¿Qué es la mega evolución?
```

El sistema responde con información de los artículos procesados, resalta en el grafo los nodos relevantes (Chandelure, Charizard, Garchomp, etc.) y muestra el concepto compartido "Mega Evolution" que los conecta.

### Datos

El sistema tiene **dos fuentes de datos**:

1. **Wikipedia**: el usuario ingresa un tema y el pipeline obtiene automáticamente el artículo principal y sus artículos relacionados mediante la **MediaWiki Action API**.
2. **Documentos del usuario**: el usuario sube archivos `.txt`, `.md` o `.pdf`, que se guardan en `data/uploads/` y se procesan con exactamente el mismo pipeline que los artículos de Wikipedia.

Ambas fuentes terminan en el mismo corpus unificado: los documentos subidos se dividen automáticamente en sub-artículos por tema detectado, de modo que un solo archivo puede generar varios nodos en el grafo.

### ¿Por qué NLP?

El producto trabaja directamente con grandes cantidades de texto y necesita interpretar las consultas del usuario para encontrar información relevante dentro de los documentos.

Una búsqueda tradicional basada únicamente en palabras clave puede fallar cuando la consulta y el contenido relevante utilizan palabras diferentes. Por ello, el sistema necesita técnicas capaces de **representar, comparar y analizar el significado del texto**, así como identificar conceptos importantes y generar explicaciones comprensibles.

NLP permite realizar estas tareas y transformar documentos no estructurados en información que pueda ser consultada de manera interactiva.

### Valor para el usuario final

El valor principal para el estudiante es **reducir el tiempo y esfuerzo necesarios para encontrar y comprender información** dentro de sus propios materiales y de fuentes externas como Wikipedia.

En lugar de navegar manualmente entre documentos para localizar conceptos relacionados, el estudiante puede utilizar una única interfaz para explorar un tema, visualizar sus relaciones y consultar en lenguaje natural.

---

## 3. Fundamento teórico

El proyecto se apoya en **semántica distribucional (TF-IDF + embeddings de spaCy)**, **reconocimiento de entidades nombradas (NER)** y un **modelo de lenguaje preentrenado local (Qwen3 vía Ollama)**.

### 3.1. Representación y similitud de texto

Los documentos se dividen en fragmentos (chunks) que se representan mediante vectores TF-IDF. La consulta del usuario también se vectoriza de la misma forma, permitiendo comparar su contenido con el de los chunks disponibles mediante **similitud coseno**.

Esta será la técnica principal utilizada para la **recuperación de información relevante** (RAG): seleccionar los chunks más parecidos a la pregunta antes de enviarlos al modelo de lenguaje.

### 3.2. Embeddings y NER con spaCy

spaCy (`en_core_web_sm`) se utiliza en dos momentos:

- **Embeddings de chunks**: cada chunk genera un vector semántico que se guarda en `chunks_embeddings.json`.
- **NER**: sobre el texto completo de cada artículo se extraen **entidades** (personas, organizaciones, lugares) y **conceptos clave**, que alimentan el análisis del grafo.

### 3.3. Modelo de lenguaje preentrenado (Ollama)

Las respuestas se generan con **Qwen3 4B** ejecutado localmente mediante Ollama (`http://localhost:11434`). El modelo opera en dos modos:

- **Corpus**: cuando la pregunta tiene chunks relevantes, responde fundamentado en ellos.
- **Conocimiento general**: cuando no hay chunks relevantes, responde con su conocimiento propio, avisando al usuario que la respuesta no proviene de los documentos procesados.

### 3.4. División automática por temas

Para los documentos subidos por el usuario, se detectan automáticamente las **frases nominales recurrentes** (noun chunks de spaCy) y se agrupa el contenido por tema, de modo que cada tema se convierte en un artículo independiente dentro del grafo.

---

## 4. Arquitectura y funcionamiento

### 4.1. Vista general del flujo

```text
                    ENTRADA DEL USUARIO
              ┌──────────────────────────┐
              │ Tema de Wikipedia        │
              │ y/o archivos .txt/.md/.pdf│
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │   server.py (HTTP :8000) │
              │  /api/process_topic      │
              │  /api/upload             │
              │  /api/process_uploads    │
              │  /api/ask                │
              │  /api/reset              │
              └────────────┬─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                                     │
┌───────▼────────┐                  ┌─────────▼──────────┐
│ pipeline.py    │                  │ process_uploads.py │
│ (tema wiki)    │                  │ (solo documentos)  │
└───────┬────────┘                  └─────────┬──────────┘
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
              data/processed/  (corpus unificado)
        transformed_articles.json
        chunks_dataframe.csv
        chunks_embeddings.json
        entities.json
                           │
              ┌────────────▼─────────────┐
              │ generate_graph_data.py   │
              │  → graph_data.json       │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │   app.html (frontend)    │
              │  vis-network + preguntas │
              └──────────────────────────┘
```

### 4.2. Adquisición de artículos de Wikipedia

Cuando el usuario ingresa un tema (por ejemplo, `Halo videogame`), `pipeline.py` ejecuta `WikipediaExtractor` (`src/data_extraction/wikipedia_extractor.py`), que usa la **MediaWiki Action API** de Wikipedia:

1. **Búsqueda del artículo principal**: se consulta la API de búsqueda para resolver el tema coloquial al artículo correcto (ej: "Halo videogame" → "Halo (franchise)").
2. **Extracción del contenido**: se obtiene el `extract` (texto plano del artículo) junto con metadatos: `page_id`, `categories`, `links`, `last_revision` y `extracted_at`.
3. **Artículos relacionados**: mediante el parámetro `morelike` de la API se obtienen hasta `max_related` artículos relacionados (por defecto 10), que ya vienen ordenados por relevancia según Wikipedia. Se respeta el *rate limit* de la API con delays entre peticiones.
4. **Limpieza HTML**: se elimina ruido del wikitexto/HTML (`.mw-editsection`, referencias, tablas de contenido, etc.).

Todo el material crudo se guarda en `data/raw/` (`main_article.json`, `related_articles.json`).

### 4.3. Procesamiento de documentos del usuario

Los archivos subidos se guardan primero en `data/uploads/` (los nombres duplicados se renombran con `_1`, `_2`...) y luego se procesan en `src/data_extraction/document_loader.py`:

1. **Extracción de texto** según el formato:
   - `.txt` / `.md`: lectura directa en UTF-8.
   - `.pdf`: extracción de texto página por página con `pypdf`.
2. **Conversión al esquema de artículo**: cada archivo se transforma en un diccionario con la misma estructura que un artículo de Wikipedia (`title`, `page_id`, `extract`, `categories`, `links`, ...), marcado con `categories: ['uploaded_document']` para distinguirlo.
3. **División por temas** (la parte más importante para el grafo):
   - El texto se divide en **unidades** (~párrafos de hasta 600 caracteres; los textos sin estructura de párrafos se agrupan por oraciones).
   - spaCy extrae **frases nominales** (noun chunks) de cada unidad.
   - Las frases que aparecen en **≥2 unidades pero no en más del 50% del documento** se consideran temas (las presentes en casi todo el texto son transversales y se excluyen).
   - Cada unidad se asigna al tema con más repeticiones; las unidades sin tema claro van a un nodo `· General`.
   - Resultado: un archivo puede generar varios artículos `mi_doc · Tema1`, `mi_doc · Tema2`, etc.

### 4.4. Transformación y chunking

`DataTransformer` (`src/transformation/data_transformation.py`) procesa cada artículo (Wikipedia o upload) con el mismo flujo:

1. **Limpieza**: espacios múltiples, caracteres especiales, referencias `[1]`, URLs, HTML residual.
2. **Normalización**: minúsculas y formas normalizadas del texto.
3. **Segmentación**: el texto limpio se divide en **chunks de ~500 caracteres con 50 de solapamiento**, respetando límites de frase.
4. **Salida**: `transformed_articles.json` (artículo completo con sus chunks y metadatos) y `chunks_dataframe.csv` (una fila por chunk: `article_title`, `article_id`, `chunk_id`, `chunk_text`, `chunk_length`).

### 4.5. Análisis semántico: embeddings y NER

`SemanticAnalyzer` (`src/embeddings/semantic_analyzer.py`) genera para cada artículo:

- **Embeddings por chunk** (`chunks_embeddings.json`): vector semántico de spaCy para cada chunk.
- **Entidades** (`entities.json`): personas, organizaciones y lugares extraídos del texto completo.
- **Conceptos clave**: frases importantes del artículo.

### 4.6. Corpus unificado

`pipeline.py` (tema Wikipedia) y `process_uploads.py` (solo documentos) escriben los mismos archivos canónicos en `data/processed/`:

- `transformed_articles.json`
- `chunks_dataframe.csv`
- `chunks_embeddings.json`
- `entities.json`

Los uploads se colocan justo después del artículo principal en la lista, para que tengan buen puntaje de relevancia en el grafo. Cuando se procesa un tema nuevo, el pipeline **reconstruye todo el corpus** e incluye automáticamente los documentos de `data/uploads/` (los uploads persisten entre temas dentro de la misma sesión del servidor). Cuando solo se suben documentos, `process_uploads.py` los **re-deriva por completo**: purga los artículos `uploaded_document` anteriores y fusiona los nuevos con los artículos de Wikipedia existentes.

**Ciclo de vida de los datos**: `server.py` elimina todo el contenido de `data/uploads/`, `data/raw/` y `data/processed/` **cada vez que arranca** — cada sesión del servidor comienza con una base de conocimiento vacía. Lo mismo hace el botón "Nueva Exploración" de la interfaz (endpoint `/api/reset`) sin necesidad de reiniciar.

### 4.7. Generación de respuestas (RAG con Ollama)

`OllamaGenerator` (`src/generation/ollama_generator.py`) responde preguntas en `/api/ask`:

1. **Recuperación TF-IDF**: la pregunta se vectoriza con el mismo `TfidfVectorizer` del corpus y se calculan similitudes coseno contra todos los chunks. Se toman los `top_k` más relevantes.
2. **Umbral de relevancia**: si la mejor similitud es < 0.12, no hay contexto útil → el modelo responde con **conocimiento general** (`source: 'general'`).
3. **Contexto compacto**: de cada chunk relevante se extraen solo las **oraciones que contienen términos de la pregunta** (eliminando referencias, citas bibliográficas y enlaces), produciendo ~900 caracteres por chunk en vez de ~3000 de ruido.
4. **Generación**: se construye un prompt en español con el contexto y la pregunta, y se envía a Ollama (`qwen3:4b-instruct-2507-q4_K_M`) con `num_predict` acotado. El modelo responde en el idioma de la pregunta.
5. **Respuesta**: devuelve `answer`, `chunks` relevantes (con similitud), `relevant_articles`, `source` (`corpus`/`general`) y entidades del chunk principal.

### 4.8. Grafo de conocimiento

`generate_graph_data.py` construye `graph_data.json` a partir de los datos procesados:

**Nodos:**

- **Artículo principal** (dorado, tamaño 16): el tema de Wikipedia consultado.
- **Artículos relacionados** (azul): los del pipeline, filtrados por un **puntaje de relevancia** = similitud TF-IDF al artículo principal × bonus por posición en el pipeline (Wikipedia ya los ordena por relevancia). Solo se conservan los ~12 mejores; el resto se descarta para evitar nodos irrelevantes.
- **Documentos subidos** (morado `#C678DD`): los artículos `uploaded_document` **siempre se conservan** (son contenido elegido explícitamente por el usuario) y aparecen como `archivo · Tema`.
- **Conceptos compartidos** (verde): frases nominales (spaCy) presentes en ≥2 artículos pero no en todos, detectadas automáticamente. Son agnósticas al dominio: el filtro usa solo palabras universales (gramática, meta/web) más detección automática de boilerplate del corpus (palabras presentes en todos los artículos). Ejemplo real: "Mega Evolution" conectando los Pokémon que la tienen.

**Aristas:**

- `main_connection`: del artículo principal a cada artículo conservado.
- `similarity`: entre artículos con similitud TF-IDF > 0.25.
- `article_concept`: de cada artículo a los conceptos que contiene.

**Visualización** (`app.html` + `vis-network`):

- El grafo usa física barnes-hut con configuración One Dark.
- Click en un nodo muestra un modal con sus chunks (limpios de referencias y citas).
- **Highlights al preguntar**: cuando `/api/ask` devuelve chunks relevantes, solo se resaltan los **nodos** de los artículos realmente relevantes. El puntaje combina la **mejor similitud del artículo** con una **bonificación por coincidencia de título** (`+0.25 × fracción de palabras del título presentes en la pregunta`), de modo que si el usuario menciona un nodo por nombre, ese nodo se prioriza aunque sus chunks no sean los mejor rankeados. Se exige un umbral relativo al mejor resultado (`max(0.25, top_score × 0.5)`) y se limita a **4 nodos máximo**. Las aristas no se modifican. La cámara hace **zoom animado** al área resaltada (`network.fit`). Los estilos originales de los nodos se guardan al crear el grafo, por lo que al hacer otra pregunta los highlights anteriores se eliminan correctamente (los colores mutados por vis-network ya no se usan para restaurar).

### 4.9. Servidor y frontend

- `server.py`: `ThreadingHTTPServer` que sirve `app.html` y expone la API REST (`/api/ask`, `/api/process_topic`, `/api/upload`, `/api/process_uploads`, `/api/reset`). El generador de Ollama es un **singleton** que se recarga automáticamente cuando cambia `chunks_dataframe.csv` (detecta por fecha de modificación). Al arrancar, el servidor **limpia todos los datos** (`data/uploads/`, `data/raw/`, `data/processed/`). Los logs incluyen timestamps por request, el progreso de los pipelines (stdout heredado de los subprocesos) y tiempo de respuesta de `/api/ask` con fuente y similitud top.
- `app.html`: interfaz estilo One Dark con pantalla de bienvenida (tema + archivos), barra de progreso del pipeline, grafo interactivo, sección de preguntas y botón "Nueva Exploración" que resetea todo el estado.

---

## 5. Estructura del proyecto

```text
├── app.html                  # Frontend: grafo + preguntas
├── server.py                 # Servidor HTTP + API REST
├── pipeline.py               # Pipeline de Wikipedia (tema → corpus)
├── process_uploads.py        # Fusiona data/uploads/ con el corpus
├── generate_graph_data.py    # Genera graph_data.json
├── main.py                   # CLI alternativo del pipeline
├── requirements.txt
├── tests/                    # Suite pytest (33 tests)
├── src/
│   ├── data_extraction/
│   │   ├── wikipedia_extractor.py   # MediaWiki API: extracción y limpieza
│   │   └── document_loader.py       # txt/md/pdf → artículos por tema
│   ├── transformation/
│   │   └── data_transformation.py   # Limpieza, normalización, chunks
│   ├── embeddings/
│   │   └── semantic_analyzer.py     # Embeddings spaCy + NER
│   └── generation/
│       └── ollama_generator.py      # TF-IDF + RAG + Ollama
├── static/vendor/vis-network.min.js
└── data/
    ├── raw/                  # Artículos crudos de Wikipedia
    ├── processed/            # Corpus unificado (se regenera)
    └── uploads/              # Documentos del usuario (se limpian al reiniciar el servidor)
```

---

## 6. Flujo end-to-end del usuario

```text
Tema + archivos ──► server.py ──► pipeline / process_uploads
                                        │
                              data/processed/ (corpus)
                                        │
                              generate_graph_data.py
                                        │
                              graph_data.json ──► app.html (grafo)
                                        │
                          Pregunta ──► /api/ask ──► OllamaGenerator
                                        │              │
                                        │        TF-IDF → chunks → prompt
                                        │              │
                                        └── respuesta + nodos relevantes
                                             (highlight + zoom)
```

---

## 7. Decisiones de diseño documentadas

- **Umbral de similitud del grafo (~0.25)**: las similitudes TF-IDF entre artículos del mismo dominio están comprimidas (0.17–0.29 por vocabulario común), por lo que un umbral alto vaciaría el grafo. Se usa en cambio un **puntaje combinado** (similitud × posición) para seleccionar artículos.
- **Conceptos compartidos agnósticos**: en lugar de listas de palabras por dominio, se detecta automáticamente el boilerplate del corpus (palabras presentes en todos los artículos) y se filtra con una lista universal de palabras de función/meta.
- **Dos modos de respuesta**: el modelo responde con el corpus cuando hay relevancia y con conocimiento general cuando no, en lugar de bloquearse — esto permite preguntar cualquier tema.
- **Uploads siempre visibles en el grafo**: aunque no sean relevantes al tema Wikipedia principal, el usuario los subió explícitamente, así que no se filtran por el puntaje de relevancia.
- **Recarga del generador por mtime**: evita datos obsoletos en memoria cuando el pipeline regenera el corpus.
- **Highlights selectivos en el grafo**: se evaluó resaltar todos los artículos de los chunks recuperados, pero con 8 chunks por respuesta casi todos los nodos se iluminaban. Se optó por quedarse solo con la mejor similitud por artículo, un umbral relativo al resultado top y un máximo de 4 nodos, de modo que el highlight sea una señal clara de relevancia.
- **Coincidencia de título en highlights**: la similitud de chunks sola fallaba cuando el usuario nombraba un nodo específico (los temas de uploads, con texto corto y denso, ganaban la recuperación). Se agregó una bonificación por palabras del título presentes en la pregunta para que el nodo mencionado se destaque.
- **Estado por sesión**: los datos se limpian al iniciar el servidor y con "Nueva Exploración", evitando corpus obsoleto entre sesiones y garantizando que cada exploración parta de cero.