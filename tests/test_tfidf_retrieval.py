"""Tests para la recuperación TF-IDF de chunks relevantes."""
import json
import pytest
from pathlib import Path

from ollama_generator import OllamaGenerator

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def generator(tmp_path):
    """Generador con datos sintéticos mínimos."""
    gen = OllamaGenerator.__new__(OllamaGenerator)
    gen.model_name = "test-model"
    gen.ollama_url = "http://localhost:11434"
    gen.chunks = []
    gen.chunk_embeddings = []
    gen.entities = {}
    gen.concepts = {}
    gen.article_titles = []
    gen.tfidf_vectorizer = None
    gen.tfidf_matrix = None

    # Crear fixtures temporales con datos sintéticos
    chunks_csv = tmp_path / "chunks.csv"
    emb_json = tmp_path / "emb.json"
    ent_json = tmp_path / "entities.json"

    chunks_csv.write_text(
        "article_title,article_id,chunk_id,chunk_text,chunk_length\n"
        'Wii,1,0,"Wii is a console by Nintendo with motion controls. It was released in 2006.",80\n'
        'Wii,1,1,"Wii Sports includes tennis and bowling games for the console.",80\n'
        'Charizard,2,0,"Charizard is a Fire and Flying type Pokemon. It evolves from Charmeleon.",90\n'
        'Chandelure,3,0,"Chandelure is a Ghost and Fire type Pokemon introduced in generation 5.",90\n',
        encoding="utf-8",
    )
    emb_json.write_text(json.dumps([
        {"embedding": [0.1] * 96},
        {"embedding": [0.1] * 96},
        {"embedding": [0.1] * 96},
        {"embedding": [0.1] * 96},
    ]), encoding="utf-8")
    ent_json.write_text(json.dumps({"entities": {}, "concepts": {}}), encoding="utf-8")

    gen.load_data(str(chunks_csv), str(emb_json), str(ent_json))
    return gen


class TestTFIDFRetrieval:
    def test_retrieves_relevant_article(self, generator):
        results = generator.retrieve_relevant_chunks("what is the Wii console", top_k=3)
        assert len(results) > 0
        # El chunk más relevante debe ser de Wii
        assert results[0][3] == "Wii"

    def test_retrieves_pokemon_article(self, generator):
        results = generator.retrieve_relevant_chunks("Charizard Pokemon type", top_k=3)
        titles = [r[3] for r in results]
        assert "Charizard" in titles

    def test_returns_similarity_scores(self, generator):
        results = generator.retrieve_relevant_chunks("Wii", top_k=2)
        for idx, sim, chunk, title in results:
            assert 0.0 <= sim <= 1.0
            assert isinstance(chunk, str)
            assert isinstance(title, str)

    def test_top_k_respected(self, generator):
        results = generator.retrieve_relevant_chunks("game console", top_k=2)
        assert len(results) <= 2

    def test_no_data_raises_error(self):
        gen = OllamaGenerator.__new__(OllamaGenerator)
        gen.tfidf_vectorizer = None
        gen.tfidf_matrix = None
        with pytest.raises(ValueError, match="TF-IDF"):
            gen.retrieve_relevant_chunks("test")
