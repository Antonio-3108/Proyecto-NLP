"""Tests para el servidor y el procesamiento de preguntas."""
import json
import pytest

import server
from server import KnowledgeHandler


class FakeGenerator:
    """Generador falso para tests sin Ollama."""
    def answer_question(self, query):
        return {
            "answer": f"Respuesta a: {query}",
            "chunks": [
                {"index": 0, "similarity": 0.8, "text": "chunk text", "article": "Test Article"}
            ],
            "concepts": ["test concept"],
            "entities": {"PERSON": ["Test Person"]},
            "top_similarity": 0.8,
            "source": "corpus",
        }


class TestProcessQuestion:
    def test_returns_answer_dict(self, monkeypatch):
        monkeypatch.setattr(server, "get_generator", lambda: FakeGenerator())
        result = KnowledgeHandler.process_question(object(), "test question")

        assert "answer" in result
        assert result["answer"] == "Respuesta a: test question"
        assert result["source"] == "corpus"
        assert "Test Article" in result["relevant_articles"]

    def test_handles_generator_error(self, monkeypatch):
        def bad_generator():
            raise RuntimeError("Ollama no disponible")

        monkeypatch.setattr(server, "get_generator", bad_generator)
        result = KnowledgeHandler.process_question(object(), "test")

        assert "error" in result
        assert "answer" in result  # siempre debe tener answer

    def test_relevant_articles_extracted(self, monkeypatch):
        monkeypatch.setattr(server, "get_generator", lambda: FakeGenerator())
        result = KnowledgeHandler.process_question(object(), "q")
        assert isinstance(result["relevant_articles"], list)
        assert len(result["relevant_articles"]) > 0


class TestJSONResponse:
    def test_send_json_response_format(self):
        """Verifica que la respuesta JSON sea válida y tenga Content-Length."""
        # Mock mínimo del handler
        class MockHandler:
            def __init__(self):
                self.headers_sent = []
                self.body = b""

            def send_response(self, code):
                self.status = code

            def send_header(self, key, value):
                self.headers_sent.append((key, value))

            def end_headers(self):
                pass

            def wfile_write(self, data):
                self.body = data

        # Simular wfile
        class MockWfile:
            def __init__(self, handler):
                self.handler = handler

            def write(self, data):
                self.handler.body = data

        handler = MockHandler()
        handler.wfile = MockWfile(handler)

        # Llamar al método real
        KnowledgeHandler.send_json_response(handler, {"test": "value", "num": 42})

        # Verificar body es JSON válido
        data = json.loads(handler.body.decode("utf-8"))
        assert data["test"] == "value"
        assert data["num"] == 42

        # Verificar headers
        headers = dict(handler.headers_sent)
        assert "Content-Length" in headers
        assert "application/json" in headers.get("Content-type", "")
        # Content-Length debe coincidir con el tamaño real
        assert int(headers["Content-Length"]) == len(handler.body)
