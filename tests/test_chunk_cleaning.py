"""Tests unitarios para la limpieza de chunks y extracción de oraciones."""
import pytest

from ollama_generator import OllamaGenerator


@pytest.fixture
def generator():
    """Instancia del generador sin __init__ (evita conexión a Ollama)."""
    gen = OllamaGenerator.__new__(OllamaGenerator)
    return gen


class TestCleanChunkText:
    def test_removes_numeric_references(self, generator):
        text = "Satoru Iwata was a Japanese businessman[1][2] who worked at Nintendo."
        result = generator._clean_chunk_text(text)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "Satoru Iwata" in result

    def test_removes_markdown_links(self, generator):
        text = "See [Nintendo](https://nintendo.com) for more info."
        result = generator._clean_chunk_text(text)
        assert "https://nintendo.com" not in result
        assert "Nintendo" in result

    def test_removes_html_tags(self, generator):
        text = "Hello <b>world</b> this is <i>text</i>."
        result = generator._clean_chunk_text(text)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "world" in result

    def test_removes_citation_sentences(self, generator):
        text = (
            "Typhlosion is a Fire-type Pokemon. "
            "It was designed by Ken Sugimori. "
            "Archived from the original on December 22, 2024. Retrieved October 26, 2024."
        )
        result = generator._clean_chunk_text(text)
        assert "Archived from the original" not in result
        assert "Retrieved" not in result
        assert "Typhlosion is a Fire-type" in result

    def test_removes_reference_sections(self, generator):
        text = (
            "Chandelure is a Pokemon species. It is the evolved form of Lampent. "
            "References [1] External links [2]"
        )
        result = generator._clean_chunk_text(text)
        # El contenido antes de 'References' se conserva
        assert "Chandelure is a Pokemon species" in result

    def test_collapses_whitespace(self, generator):
        text = "Hello    world\n\n  this   is   text"
        result = generator._clean_chunk_text(text)
        assert "  " not in result
        assert result == result.strip()


class TestExtractRelevantSentences:
    def test_extracts_matching_sentences(self, generator):
        chunk = (
            "The Wii is a home video game console developed by Nintendo. "
            "It was released in 2006. "
            "The console uses motion controls called the Wii Remote. "
            "Sales exceeded 100 million units worldwide."
        )
        result = generator._extract_relevant_sentences(chunk, "what is the Wii Remote", max_chars=500)
        assert "Wii Remote" in result

    def test_respects_max_chars(self, generator):
        chunk = "Sentence one about Wii. " * 50
        result = generator._extract_relevant_sentences(chunk, "Wii console", max_chars=200)
        assert len(result) <= 200 + 100  # margen por frase completa

    def test_fallback_to_chunk_start_when_no_match(self, generator):
        chunk = "x" * 500 + " totally unrelated content here."
        result = generator._extract_relevant_sentences(chunk, "zzzznomatch", max_chars=300)
        # Sin matches: devuelve inicio del chunk
        assert len(result) <= 300
        assert result.startswith("x")

    def test_empty_chunk(self, generator):
        result = generator._extract_relevant_sentences("", "test query", max_chars=100)
        assert result == ""
