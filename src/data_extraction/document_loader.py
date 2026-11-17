"""
Cargador de documentos subidos por el usuario (txt, md, pdf).
Convierte archivos locales al mismo formato de artículo que usa el pipeline
de Wikipedia, para que se procesen de igual manera.

Además, divide cada documento en sub-artículos por tema detectado
automáticamente (frases nominales recurrentes vía spaCy), de modo que un
mismo documento genera varios nodos en el grafo de conocimiento.
"""

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}

# Palabras genéricas/de función que no sirven como tema
_GENERIC_WORDS = {
    # gramática inglesa
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'by',
    'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it',
    'its', 'this', 'that', 'these', 'those', 'there', 'their', 'they', 'them',
    'he', 'she', 'his', 'her', 'we', 'our', 'you', 'your', 'i', 'me', 'my',
    'not', 'but', 'if', 'then', 'than', 'so', 'such', 'no', 'yes', 'can',
    'could', 'will', 'would', 'should', 'may', 'might', 'must', 'do', 'does',
    'did', 'have', 'has', 'had', 'one', 'two', 'also', 'just', 'about', 'into',
    'over', 'under', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'up', 'down', 'out', 'off', 'again', 'once', 'here', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'only', 'own', 'same', 'very', 'too', 'who', 'whom',
    'which', 'what', 'while', 'because', 'until', 'against', 'among',
    # meta / web / citas
    'article', 'wikipedia', 'page', 'section', 'chapter', 'figure', 'table',
    'image', 'file', 'edit', 'view', 'history', 'talk', 'read', 'external',
    'link', 'links', 'reference', 'references', 'source', 'sources', 'citation',
    'cite', 'cited', 'retrieved', 'archived', 'original', 'accessed', 'available',
    'online', 'website', 'copyright', 'license', 'creative', 'commons', 'isbn',
    'doi', 'url', 'http', 'https', 'org', 'com', 'www', 'per', 'via',
    # tiempo
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'year', 'years',
    'month', 'months', 'day', 'days', 'time', 'times', 'century', 'centuries',
}

_nlp = None


def _get_nlp():
    """Carga perezosa del modelo spaCy (se usa para detectar temas)."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_text_from_file(file_path) -> str:
    """
    Extrae el texto plano de un archivo según su extensión.

    Args:
        file_path: Ruta al archivo

    Returns:
        Texto extraído

    Raises:
        ValueError: Si la extensión no es soportada
        RuntimeError: Si falta una dependencia (pypdf)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in ('.txt', '.md'):
        return path.read_text(encoding='utf-8', errors='replace')

    if ext == '.pdf':
        try:
            import pypdf
        except ImportError:
            raise RuntimeError(
                "Para leer PDFs instala pypdf: pip install pypdf"
            )
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or '' for page in reader.pages]
        return '\n\n'.join(pages)

    raise ValueError(f"Extensión no soportada: {ext}. Usa: {', '.join(SUPPORTED_EXTENSIONS)}")


def _article_dict(title: str, page_id: int, extract: str) -> Dict:
    """Construye el diccionario con el esquema de artículo del pipeline."""
    return {
        'title': title,
        'page_id': page_id,
        'extract': extract,
        'categories': ['uploaded_document'],
        'links': [],
        'last_revision': '',
        'extracted_at': datetime.now(timezone.utc).isoformat()
    }


def _stable_id(seed: str) -> int:
    return int(hashlib.md5(seed.encode('utf-8')).hexdigest()[:8], 16)


def document_to_article(file_path) -> Dict:
    """
    Convierte un archivo subido al formato de artículo del pipeline
    (documento completo, sin dividir por temas).

    Args:
        file_path: Ruta al archivo

    Returns:
        Diccionario con la misma estructura que un artículo de Wikipedia
    """
    path = Path(file_path)
    text = extract_text_from_file(path)

    if not text.strip():
        raise ValueError(f"El archivo '{path.name}' no contiene texto extraíble")

    return _article_dict(path.stem, _stable_id(str(path.resolve())), text)


# ---------------------------------------------------------------------------
# División por temas
# ---------------------------------------------------------------------------

def _group_sentences(text: str, max_chars: int = 600) -> List[str]:
    """Agrupa oraciones en unidades de ~max_chars respetando límites de frase."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    units = []
    current = ''
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            units.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        units.append(current)
    return units


def _split_into_units(text: str, max_chars: int = 600) -> List[str]:
    """
    Divide el texto en unidades temáticas.

    Prioriza los párrafos (separados por líneas en blanco): cada párrafo es
    una unidad, y los párrafos largos se subdividen por oraciones. Si el
    texto no tiene estructura de párrafos (txt/pdf corrido), se agrupan
    oraciones en ventanas de ~max_chars.
    """
    # Quitar marcadores de encabezado markdown
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Párrafos = bloques separados por líneas en blanco
    blocks = [b.strip() for b in re.split(r'\n\s*\n+', text) if b.strip()]

    if len(blocks) <= 1:
        # Sin estructura de párrafos: unir líneas sueltas y agrupar oraciones
        flat = ' '.join(line.strip() for line in text.split('\n') if line.strip())
        return _group_sentences(flat, max_chars)

    # Los bloques diminutos (típicamente encabezados) se anexan al siguiente
    paragraphs = [
        ' '.join(line.strip() for line in block.split('\n') if line.strip())
        for block in blocks
    ]
    merged_blocks = []
    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]
        if len(paragraph) < 40 and i + 1 < len(paragraphs):
            paragraph = f"{paragraph} {paragraphs[i + 1]}"
            i += 2
        else:
            i += 1
        merged_blocks.append(paragraph)

    units = []
    for paragraph in merged_blocks:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(_group_sentences(paragraph, max_chars))
    return units


def _extract_topic_phrases(text: str, nlp) -> Dict[str, str]:
    """
    Extrae frases nominales candidatas a tema de una unidad de texto.

    Returns:
        Lista de (clave_normalizada, forma_superficial) por cada frase encontrada
    """
    doc = nlp(text[:2500])
    phrases = []

    for chunk in doc.noun_chunks:
        toks = [
            t for t in chunk
            if not t.is_punct and not t.is_stop
            and t.text.lower() not in _GENERIC_WORDS
        ]
        if not toks:
            continue

        words = [t.lemma_.lower() for t in toks]
        if not (1 <= len(words) <= 4):
            continue
        # Debe tener al menos una palabra de contenido (>=4 letras, no genérica)
        if not any(len(w) >= 4 and w not in _GENERIC_WORDS for w in words):
            continue

        key = ' '.join(words)
        surface = ' '.join(t.text for t in toks)
        phrases.append((key, surface))

    return phrases


def _group_units_by_topic(units: List[str], max_topics: int = 8) -> Optional[List[Tuple[str, str]]]:
    """
    Detecta los temas principales del documento y agrupa las unidades.

    Args:
        units: Unidades de texto del documento
        max_topics: Máximo de temas (nodos) por documento

    Returns:
        Lista de (etiqueta, texto) por tema, o None si no hay división útil
    """
    if len(units) < 3:
        return None

    nlp = _get_nlp()

    # Frases por unidad: Counter de claves (con repeticiones) + superficies
    unit_counters = []
    surfaces = {}
    df = Counter()
    total_counts = Counter()

    for u in units:
        pairs = _extract_topic_phrases(u, nlp)
        counter = Counter(key for key, _ in pairs)
        unit_counters.append(counter)
        for key, surface in pairs:
            surfaces.setdefault(key, surface)
            total_counts[key] += 1
        for key in counter:
            df[key] += 1

    n_units = len(units)
    # Candidatos: aparecen en >=2 unidades pero no en la mayoría del documento
    # (un término presente en casi todo el texto es transversal, no un subtema)
    candidates = [
        k for k, c in df.items()
        if 2 <= c < max(3, int(0.5 * n_units))
    ]
    if not candidates:
        return None

    # Ordenar por df desc, luego por frecuencia total desc
    candidates.sort(key=lambda k: (df[k], total_counts[k]), reverse=True)

    # Eliminar temas que son subconjunto de otro tema mejor rankeado
    # (ej: "evolution" cede ante "mega evolution")
    kept_keys = []
    kept_wordsets = []
    for key in candidates:
        words = set(key.split())
        if any(words < ws for ws in kept_wordsets):
            continue
        kept_keys.append(key)
        kept_wordsets.append(words)
        if len(kept_keys) >= max_topics:
            break

    if not kept_keys:
        return None

    topics = kept_keys

    # Asignar cada unidad a su mejor tema según repeticiones (o 'General')
    topic_units = defaultdict(list)
    general_units = []

    for i, counter in enumerate(unit_counters):
        matched = [(counter[t], len(t.split()), -topics.index(t), t)
                   for t in topics if counter.get(t, 0) > 0]
        if matched:
            # Más repeticiones > frase más larga > mejor rank
            best = max(matched)[3]
            topic_units[best].append(i)
        else:
            general_units.append(i)

    # Temas con muy poco contenido (<2 unidades) se disuelven en General
    sections = []
    dissolved = []
    for topic in topics:
        idxs = topic_units.get(topic, [])
        if len(idxs) >= 2:
            sections.append((topic, idxs))
        else:
            dissolved.extend(idxs)
    general_units.extend(dissolved)

    # Si solo queda un tema y no hay general, no vale la pena dividir
    if len(sections) <= 1 and not general_units:
        return None

    # Construir textos por sección
    result = []
    for topic, idxs in sorted(sections, key=lambda s: -len(s[1])):
        text = '\n\n'.join(units[i] for i in sorted(idxs))
        result.append((surfaces[topic], text))

    if general_units:
        text = '\n\n'.join(units[i] for i in sorted(general_units))
        result.append(('General', text))

    return result if result else None


def document_to_articles(file_path, max_topics: int = 8) -> List[Dict]:
    """
    Convierte un archivo subido en uno o más artículos (sub-divididos por tema).

    Si el documento es muy corto o no tiene temas recurrentes, devuelve
    un único artículo con el documento completo.

    Args:
        file_path: Ruta al archivo
        max_topics: Máximo de temas/nodos por documento

    Returns:
        Lista de artículos en formato del pipeline
    """
    path = Path(file_path)
    text = extract_text_from_file(path)

    if not text.strip():
        raise ValueError(f"El archivo '{path.name}' no contiene texto extraíble")

    units = _split_into_units(text)
    sections = _group_units_by_topic(units, max_topics=max_topics)

    if not sections:
        return [_article_dict(path.stem, _stable_id(str(path.resolve())), text)]

    articles = []
    for label, section_text in sections:
        title = f"{path.stem} · {label}"
        page_id = _stable_id(f"{path.resolve()}::{label}")
        articles.append(_article_dict(title, page_id, section_text))

    return articles


def load_uploaded_documents(uploads_dir) -> List[Dict]:
    """
    Carga todos los documentos soportados de un directorio, dividiendo
    cada uno en sub-artículos por tema detectado.

    Args:
        uploads_dir: Directorio con los archivos subidos

    Returns:
        Lista de artículos en formato del pipeline
    """
    uploads_path = Path(uploads_dir)
    articles = []

    if not uploads_path.exists():
        return articles

    for file_path in sorted(uploads_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            doc_articles = document_to_articles(file_path)
            articles.extend(doc_articles)
            print(f"  Documento cargado: {file_path.name} -> {len(doc_articles)} tema(s)")
        except Exception as e:
            print(f"  ERROR cargando {file_path.name}: {e}")

    return articles
