# Proyecto Final — Explorador Interactivo de Conocimiento

## 1. Idea

### Producto

Se desarrollará un **Explorador Interactivo de Conocimiento** en formato CLI/TUI orientado al estudio. La aplicación permitirá que un estudiante cargue material de estudio, como PDFs, apuntes o documentos de texto, y posteriormente consulte su contenido mediante preguntas formuladas en lenguaje natural.

Ante cada consulta, el sistema buscará la información más relevante dentro del material disponible, generará un resumen breve y mostrará los principales conceptos o entidades relacionados con la respuesta. El usuario podrá utilizar estos conceptos para formular nuevas preguntas y continuar explorando el contenido de manera interactiva.

El objetivo no es sustituir el material de estudio, sino **convertirlo en una forma más sencilla de consultar, comprender y explorar**.

### Usuario final

**Estudiantes** que necesitan comprender un tema nuevo o repasar material de estudio y desean encontrar información relevante sin tener que revisar manualmente grandes cantidades de contenido.

### Problema que resuelve

Los estudiantes pueden disponer de múltiples documentos, apuntes o materiales de clase, pero encontrar rápidamente una explicación concreta y entender qué conceptos están relacionados puede requerir revisar una gran cantidad de información.

El producto busca reducir este esfuerzo permitiendo que el estudiante consulte directamente el material mediante lenguaje natural, obtenga una explicación resumida y descubra conceptos relacionados que puede utilizar para continuar explorando el tema.

---

## 2. Justificación

### ¿Por qué vale la pena?

Los estudiantes tienen acceso a una gran cantidad de información, pero encontrar lo relevante y relacionar los conceptos puede requerir revisar manualmente múltiples documentos. El producto busca **reducir ese esfuerzo y facilitar una exploración más rápida y accesible del material de estudio**.

Por ejemplo, un estudiante podría cargar sus materiales de una clase de Natural Language Processing y preguntar:

```text
> ¿Qué es un Transformer?
```

El sistema podría presentar una explicación breve y conceptos relacionados como:

```text
• Attention
• Encoder
• Decoder
• BERT
```

El estudiante puede utilizar cualquiera de estos conceptos como punto de partida para realizar una nueva consulta.

### Datos

El MVP utilizará como fuente principal **documentos proporcionados por el usuario**, principalmente archivos PDF o texto relacionados con un tema académico.

El corpus de cada sesión se construirá a partir de estos documentos mediante un proceso de:

- extracción del texto;
- limpieza y normalización;
- eliminación de ruido;
- segmentación en fragmentos;
- eliminación o tratamiento de contenido duplicado.

Para el desarrollo y las pruebas iniciales se utilizará como dominio de referencia **Natural Language Processing y Machine Learning**, utilizando materiales relacionados con conceptos como:

- Tokenización
- TF-IDF
- Word2Vec
- Embeddings
- NER
- RNN/LSTM
- Attention
- Transformers
- BERT
- GPT
- Clasificación de texto

Como posible extensión, el sistema podrá incorporar fuentes externas como Wikipedia mediante la **MediaWiki Action API**, permitiendo ampliar el conocimiento disponible más allá de los documentos proporcionados directamente por el usuario.

### ¿Por qué NLP?

El producto trabaja directamente con grandes cantidades de texto y necesita interpretar las consultas del usuario para encontrar información relevante dentro de los documentos.

Una búsqueda tradicional basada únicamente en palabras clave puede fallar cuando la consulta y el contenido relevante utilizan palabras diferentes. Por ello, el sistema necesita técnicas capaces de **representar, comparar y analizar el significado del texto**, así como identificar conceptos importantes y generar explicaciones comprensibles.

NLP permite realizar estas tareas y transformar documentos no estructurados en información que pueda ser consultada de manera interactiva.

### Valor para el usuario final

El valor principal para el estudiante es **reducir el tiempo y esfuerzo necesarios para encontrar y comprender información** dentro de sus propios materiales.

En lugar de navegar manualmente entre documentos para localizar conceptos relacionados, el estudiante puede utilizar una única interfaz para preguntar, recibir una explicación y continuar explorando el contenido.

---

## 3. Fundamento teórico

El proyecto se apoyará principalmente en **semántica distribucional y embeddings**, complementados por **reconocimiento de entidades nombradas (NER)** y **modelos preentrenados**.

Cada técnica tendrá una función específica dentro del producto.

### 3.1. Semántica distribucional y embeddings

Los documentos se dividirán en fragmentos que serán representados mediante vectores. La consulta del usuario también podrá representarse de forma que sea posible comparar su significado con el de los fragmentos disponibles.

La similitud entre estas representaciones permitirá recuperar contenido relevante incluso cuando la consulta y el documento no utilicen exactamente las mismas palabras.

Esta será la técnica principal utilizada para implementar la **búsqueda semántica** dentro del material de estudio.

### 3.2. Reconocimiento de entidades nombradas (NER)

NER se utilizará para identificar elementos importantes dentro del contenido recuperado.

Dependiendo del dominio, estos elementos pueden incluir personas, organizaciones, lugares u otras entidades relevantes.

Las entidades identificadas podrán presentarse como **conceptos relacionados** en la respuesta, proporcionando al estudiante puntos de referencia para continuar su exploración.

### 3.3. Modelos preentrenados

Se podrá utilizar un modelo preentrenado para procesar la información recuperada y generar un **resumen breve y comprensible**.

El objetivo no será entrenar un modelo de lenguaje desde cero, sino aprovechar un modelo existente para transformar la información relevante recuperada en una respuesta útil para el estudiante.

La elección concreta del modelo y del método de generación se determinará durante el desarrollo del proyecto, conforme se cubran estos contenidos en el curso.

### 3.4. POS como técnica complementaria

El etiquetado gramatical (POS) podría utilizarse como una técnica adicional para analizar la estructura lingüística de los textos.

Sin embargo, **no será un componente obligatorio del MVP**, ya que las funcionalidades principales del producto pueden implementarse sin depender de esta técnica.

---

## 4. Flujo end-to-end

El producto recorrerá el ciclo completo solicitado en el proyecto:

```text
                  MATERIAL DE ESTUDIO
                   PDFs / apuntes / TXT
                          ↓
                CAPTURA DE INFORMACIÓN
                  Cargar documentos
                          ↓
                LIMPIEZA Y PREPARACIÓN
          Extracción, normalización, segmentación
                    y eliminación de ruido
                          ↓
                  MODELADO / PLN
          Embeddings + similitud semántica
                    NER + modelo preentrenado
                          ↓
                  CONSULTA DEL USUARIO
                Pregunta en lenguaje natural
                          ↓
              RECUPERACIÓN DE INFORMACIÓN
              Selección de contenido relevante
                          ↓
                    PROCESAMIENTO
               Resumen + conceptos relacionados
                          ↓
                    PRODUCTO FINAL
                       CLI / TUI
                          ↓
                    NUEVA CONSULTA
               El usuario continúa explorando
```

Las decisiones tomadas durante la limpieza y preparación de los documentos serán documentadas para justificar el tratamiento del ruido, duplicados, segmentación y otros cambios realizados sobre los datos.

---

## 5. Interfaz y experiencia de usuario

La aplicación tendrá una interfaz de terminal interactiva que permita al estudiante consultar su material de manera sencilla.

Ejemplo:

```text
Explora un tema:
Escribe "exit" para salir

> Qué es un Transformer?

⠿ Buscando información...

╭─╭─────────────────────────────────────────╮
│ │ # TRANSFORMER                           │
│ │ Un Transformer es una arquitectura de   │
│ │ redes neuronales utilizada ampliamente  │
│ │ en procesamiento de lenguaje natural... │
│ │                                         │
│ │ Conceptos relacionados:                 │
│ │ • Encoder                               │
│ │ • Decoder                               │
│ │                   ...                   │
╰─╰─────────────────────────────────────────╯
 ↑/↓: Continuar con el tema elegido    ←/→: Ir a busqueda anterior
 g: Ir a fuente   /: Nueva busqueda
```

La interfaz permitirá realizar nuevas consultas a partir del contenido recuperado, manteniendo el objetivo principal de **explorar y comprender el material de estudio**.

---

## 6. Alcance inicial (MVP)

Para mantener el proyecto viable durante el semestre, el MVP incluirá:

1. Carga de documentos de estudio en formatos manejables, principalmente PDF o texto.
2. Extracción y limpieza del contenido.
3. Segmentación de los documentos en fragmentos.
4. Recuperación de información relevante ante una consulta.
5. Uso de embeddings y similitud para relacionar la consulta con el contenido.
6. Identificación de conceptos o entidades relevantes mediante NER.
7. Generación de un resumen breve del contenido recuperado.
8. Presentación de la respuesta en una CLI/TUI interactiva.
9. Posibilidad de realizar nuevas consultas sobre el mismo material.

### Dominio inicial

Para el desarrollo y evaluación del MVP, el sistema se enfocará inicialmente en **Natural Language Processing y Machine Learning**. Esto permite trabajar con un conjunto de documentos controlado y directamente relacionado con los contenidos del curso.

### Posibles extensiones futuras

Una vez validado el MVP, el producto podría ampliarse para incorporar:

- Wikipedia y otras fuentes externas;
- múltiples materias o cursos;
- diferentes tipos de documentos;
- historial de exploración;
- perfiles o sesiones de estudio;
- visualización de relaciones entre conceptos;
- personalización según el nivel del estudiante;
- otras áreas de conocimiento.

Estas funcionalidades **no forman parte del alcance mínimo del MVP** y se consideran posibles extensiones del producto.