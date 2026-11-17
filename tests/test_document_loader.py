"""Tests para src/data_extraction/document_loader.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_extraction"))

from document_loader import (
    document_to_article,
    document_to_articles,
    extract_text_from_file,
    load_uploaded_documents,
    SUPPORTED_EXTENSIONS,
)


@pytest.fixture
def sample_text():
    return "Halo es una franquicia de videojuegos. " * 20


def test_extract_text_txt(tmp_path, sample_text):
    f = tmp_path / "doc.txt"
    f.write_text(sample_text, encoding='utf-8')
    assert extract_text_from_file(f) == sample_text


def test_extract_text_md(tmp_path, sample_text):
    f = tmp_path / "doc.md"
    f.write_text(f"# Título\n\n{sample_text}", encoding='utf-8')
    text = extract_text_from_file(f)
    assert "Título" in text
    assert "Halo" in text


def test_extract_text_unsupported(tmp_path):
    f = tmp_path / "doc.exe"
    f.write_bytes(b"binary")
    with pytest.raises(ValueError, match="no soportada"):
        extract_text_from_file(f)


def test_document_to_article_schema(tmp_path, sample_text):
    f = tmp_path / "mi_doc.txt"
    f.write_text(sample_text, encoding='utf-8')
    article = document_to_article(f)

    # Debe tener el mismo esquema que un artículo de Wikipedia
    for key in ('title', 'page_id', 'extract', 'categories', 'links',
                'last_revision', 'extracted_at'):
        assert key in article

    assert article['title'] == 'mi_doc'
    assert article['extract'] == sample_text
    assert isinstance(article['page_id'], int)
    assert article['categories'] == ['uploaded_document']
    assert article['links'] == []


def test_document_to_article_empty_file(tmp_path):
    f = tmp_path / "vacio.txt"
    f.write_text("   \n  ", encoding='utf-8')
    with pytest.raises(ValueError, match="no contiene texto"):
        document_to_article(f)


def test_load_uploaded_documents_filters(tmp_path, sample_text):
    (tmp_path / "a.txt").write_text(sample_text, encoding='utf-8')
    (tmp_path / "b.md").write_text(sample_text, encoding='utf-8')
    (tmp_path / "c.exe").write_bytes(b"no")
    (tmp_path / "d.pdf").write_bytes(b"%PDF-fake")

    articles = load_uploaded_documents(tmp_path)
    titles = {a['title'] for a in articles}

    # txt y md se cargan; exe se omite; pdf falla al extraer (no es PDF válido)
    assert 'a' in titles
    assert 'b' in titles
    assert 'c' not in titles
    assert 'd' not in titles
    assert len(articles) == 2


def test_load_uploaded_documents_missing_dir(tmp_path):
    assert load_uploaded_documents(tmp_path / "no_existe") == []


def test_supported_extensions():
    assert '.txt' in SUPPORTED_EXTENSIONS
    assert '.md' in SUPPORTED_EXTENSIONS
    assert '.pdf' in SUPPORTED_EXTENSIONS


MULTI_TOPIC_TEXT = """
Master Chief is a SPARTAN-II soldier. He fights the Covenant across many battles.
The Covenant is an alien alliance led by the Prophet of Truth.
Covenant ships include the CCS-class battlecruiser and Banshee flyers.

The Flood is a parasitic alien race. Infection forms consume humanoid hosts.
Flood colonies build hardlight constructions across installations.
The Flood was created by the Forerunners as a weapon.

The UNSC is humanity's interstellar military force. It operates the Infinity
supercarrier and maintains orbital defense platforms across the colonies.
UNSC Marines fight alongside Spartan soldiers in every major engagement.
The UNSC Navy patrols the outer regions of human space.
""" * 2


def test_document_to_articles_splits_by_topic(tmp_path):
    f = tmp_path / "guia.txt"
    f.write_text(MULTI_TOPIC_TEXT, encoding='utf-8')

    articles = document_to_articles(f)

    # Un documento multi-tema debe generar varios sub-artículos
    assert len(articles) > 1
    # Los títulos llevan el nombre del doc + el tema detectado
    assert all('·' in a['title'] for a in articles)
    assert all(a['title'].startswith('guia') for a in articles)
    # Todos comparten el esquema de artículo
    for a in articles:
        assert 'extract' in a and 'categories' in a
        assert 'uploaded_document' in a['categories']


def test_document_to_articles_short_doc_single(tmp_path, sample_text):
    f = tmp_path / "corto.txt"
    f.write_text(sample_text, encoding='utf-8')
    articles = document_to_articles(f)
    assert len(articles) == 1
    assert articles[0]['title'] == 'corto'
