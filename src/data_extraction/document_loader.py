"""
Cargador de documentos subidos por el usuario (txt, md, pdf).
Convierte archivos locales al mismo formato de artículo que usa el pipeline
de Wikipedia, para que se procesen de igual manera.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}


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


def document_to_article(file_path) -> Dict:
    """
    Convierte un archivo subido al formato de artículo del pipeline.

    Args:
        file_path: Ruta al archivo

    Returns:
        Diccionario con la misma estructura que un artículo de Wikipedia
    """
    path = Path(file_path)
    text = extract_text_from_file(path)

    if not text.strip():
        raise ValueError(f"El archivo '{path.name}' no contiene texto extraíble")

    # Título = nombre del archivo sin extensión
    title = path.stem

    # ID sintético estable basado en la ruta absoluta
    page_id = int(hashlib.md5(str(path.resolve()).encode('utf-8')).hexdigest()[:8], 16)

    return {
        'title': title,
        'page_id': page_id,
        'extract': text,
        'categories': ['uploaded_document'],
        'links': [],
        'last_revision': '',
        'extracted_at': datetime.now(timezone.utc).isoformat()
    }


def load_uploaded_documents(uploads_dir) -> List[Dict]:
    """
    Carga todos los documentos soportados de un directorio.

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
            article = document_to_article(file_path)
            articles.append(article)
            print(f"  Documento cargado: {file_path.name} ({len(article['extract'])} chars)")
        except Exception as e:
            print(f"  ERROR cargando {file_path.name}: {e}")

    return articles
