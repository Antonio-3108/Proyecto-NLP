"""
Módulo para embeddings semánticos y Named Entity Recognition (NER).
Usa spaCy para generar vectores de palabras y extraer entidades.
"""

import spacy
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SemanticAnalyzer:
    """Clase para análisis semántico con embeddings y NER."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Inicializa el analizador semántico.

        Args:
            model_name: Nombre del modelo de spaCy (en_core_web_sm o en_core_web_md)
        """
        try:
            self.nlp = spacy.load(model_name)
            self.use_spacy = True
            print(f"Modelo spaCy cargado: {model_name}")
        except OSError:
            print(f"Modelo {model_name} no encontrado. Intentando con en_core_web_sm...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
                print("Modelo spaCy cargado: en_core_web_sm")
            except OSError:
                print("ERROR: No se pudo cargar ningún modelo de spaCy.")
                print("Ejecuta: python -m spacy download en_core_web_sm")
                self.nlp = None
                self.use_spacy = False

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera un embedding para un texto completo.

        Args:
            text: Texto a vectorizar

        Returns:
            Vector numpy con el embedding del texto
        """
        if not self.use_spacy or not self.nlp:
            return np.zeros(96)  # Vector de ceros si no hay modelo (sm tiene 96 dims)

        doc = self.nlp(text)
        # Promedio de los vectores de las palabras
        vectors = [token.vector for token in doc if token.has_vector]
        if not vectors:
            return np.zeros(96)

        embedding = np.mean(vectors, axis=0)
        return embedding

    def get_chunk_embedding(self, chunk_text: str) -> np.ndarray:
        """
        Genera un embedding para un chunk de texto.

        Args:
            chunk_text: Texto del chunk

        Returns:
            Vector numpy con el embedding del chunk
        """
        return self.get_text_embedding(chunk_text)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud coseno entre dos textos.

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Valor de similitud entre 0 y 1
        """
        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)

        # Calcular similitud coseno
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def get_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calcula la similitud entre dos títulos de artículos.

        Args:
            title1: Primer título
            title2: Segundo título

        Returns:
            Valor de similitud entre 0 y 1
        """
        return self.calculate_similarity(title1, title2)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades nombradas de un texto usando NER.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario con tipos de entidades y sus valores
        """
        if not self.use_spacy or not self.nlp:
            return {}

        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-political entity
            'NORP': [],  # Nationalities or religious or political groups
            'FAC': [],  # Facilities
            'PRODUCT': [],
            'EVENT': [],
            'WORK_OF_ART': [],
            'LAW': [],
            'LANGUAGE': [],
            'DATE': [],
            'TIME': [],
            'MONEY': [],
            'PERCENT': [],
            'QUANTITY': [],
            'ORDINAL': [],
            'CARDINAL': []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Eliminar duplicados manteniendo orden
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return {k: v for k, v in entities.items() if v}

    def extract_key_concepts(self, text: str) -> List[str]:
        """
        Extrae conceptos clave de un texto (sustantivos y términos importantes).

        Args:
            text: Texto a analizar

        Returns:
            Lista de conceptos clave
        """
        if not self.use_spacy or not self.nlp:
            return []

        doc = self.nlp(text)

        # Extraer sustantivos y términos técnicos
        concepts = []
        for token in doc:
            # Incluir sustantivos, verbos y adjetivos importantes
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                # Filtrar tokens muy cortos o muy comunes
                if len(token.text) > 3 and token.text.lower() not in ['the', 'and', 'for', 'with']:
                    concepts.append(token.text)

        # Eliminar duplicados manteniendo orden
        concepts = list(dict.fromkeys(concepts))

        return concepts[:20]  # Limitar a 20 conceptos

    def rank_chunks_by_relevance(self, query: str, chunks: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Rankea chunks por relevancia a una consulta.

        Args:
            query: Consulta del usuario
            chunks: Lista de chunks de texto
            top_k: Número de chunks a retornar

        Returns:
            Lista de tuplas (índice, similitud, chunk)
        """
        query_embedding = self.get_text_embedding(query)

        similarities = []
        for i, chunk in enumerate(chunks):
            chunk_embedding = self.get_chunk_embedding(chunk)
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarities.append((i, float(similarity), chunk))

        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


# Función de prueba
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    # Test de embeddings
    text1 = "Natural language processing is a field of AI"
    text2 = "NLP is about understanding human language"
    text3 = "Video games are interactive entertainment"

    print("\n=== Test de Similitud ===")
    print(f"Similitud text1-text2: {analyzer.calculate_similarity(text1, text2):.3f}")
    print(f"Similitud text1-text3: {analyzer.calculate_similarity(text1, text3):.3f}")

    # Test de NER
    print("\n=== Test de NER ===")
    text = "Shigeru Miyamoto is a Japanese video game designer who works at Nintendo."
    entities = analyzer.extract_entities(text)
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {entity_list}")

    # Test de conceptos
    print("\n=== Test de Conceptos ===")
    concepts = analyzer.extract_key_concepts(text)
    print(f"Conceptos: {concepts}")
