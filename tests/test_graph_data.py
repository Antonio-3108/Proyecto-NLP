"""Tests para la generación de datos del grafo."""
import json
import pytest
from pathlib import Path

import generate_graph_data as gd


class TestGraphDataGeneration:
    def test_extract_shared_concepts_finds_common_phrases(self):
        """Frases compartidas por 2+ artículos deben detectarse."""
        texts = {
            "Article A": "Physicists study quantum entanglement in particles every day. " * 5,
            "Article B": "The experiment demonstrated quantum entanglement between photons. " * 5,
            "Article C": "Completely unrelated content about cooking recipes and pasta. " * 5,
        }
        concepts = gd.extract_shared_concepts(texts, min_docs=2, max_concepts=5)
        # Debe encontrar al menos un concepto compartido entre A y B
        assert len(concepts) >= 1
        # Todos los conceptos deben tener >= 2 artículos
        for phrase, members in concepts.items():
            assert len(members) >= 2

    def test_extract_shared_concepts_respects_min_docs(self):
        """Frases en un solo artículo no deben ser conceptos."""
        texts = {
            "Only Article": "Unique phrase appears nowhere else at all.",
        }
        # Con 1 solo documento no hay conceptos posibles (min_docs=2)
        concepts = gd.extract_shared_concepts(texts, min_docs=2, max_concepts=5)
        assert concepts == {}

    def test_extract_shared_concepts_max_limit(self):
        texts = {
            "A": "alpha beta gamma delta epsilon zeta eta theta",
            "B": "alpha beta gamma delta epsilon zeta eta theta",
            "C": "alpha beta gamma delta epsilon zeta eta theta",
        }
        concepts = gd.extract_shared_concepts(texts, min_docs=2, max_concepts=2)
        assert len(concepts) <= 2


class TestGraphDataStructure:
    def test_generate_graph_data_creates_valid_json(self, tmp_path):
        """El JSON generado debe tener la estructura esperada."""
        # Crear fixtures mínimos
        chunks_csv = tmp_path / "chunks.csv"
        entities_json = tmp_path / "entities.json"
        transformed_json = tmp_path / "transformed.json"
        output_json = tmp_path / "graph.json"

        chunks_csv.write_text(
            "article_title,article_id,chunk_id,chunk_text,chunk_length\n"
            "Main Article,1,0,Main article about a specific topic with details.,50\n"
            "Main Article,1,1,More content about the main topic here.,50\n"
            "Related Article,2,0,Related content sharing the topic vocabulary.,50\n",
            encoding="utf-8",
        )
        entities_json.write_text(json.dumps({"entities": {}, "concepts": {}}), encoding="utf-8")
        transformed_json.write_text(json.dumps([
            {"title": "Main Article"},
            {"title": "Related Article"},
        ]), encoding="utf-8")

        gd.generate_graph_data(
            str(chunks_csv), str(entities_json), str(transformed_json),
            str(output_json), max_nodes=20, max_edges=20
        )

        data = json.loads(output_json.read_text(encoding="utf-8"))
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 1

        # El primer nodo debe ser el artículo principal
        main_node = data["nodes"][0]
        assert main_node["label"] == "Main Article"
        assert main_node["color"] == "#FFD700"

        # Todas las aristas deben referenciar nodos existentes
        node_ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert edge["from"] in node_ids
            assert edge["to"] in node_ids
