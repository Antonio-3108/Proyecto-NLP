"""
Módulo para generación de resúmenes y respuestas usando Ollama.
Ollama permite ejecutar modelos de lenguaje localmente de forma fácil.
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class OllamaGenerator:
    """Generador de resúmenes y respuestas usando Ollama."""

    def __init__(self, model_name: str = "qwen3:4b-instruct-2507-q4_K_M", ollama_url: str = "http://localhost:11434"):
        """
        Inicializa el generador con Ollama.

        Args:
            model_name: Nombre del modelo de Ollama (ej: 'llama3.2:3b', 'mistral:7b')
            ollama_url: URL de la API de Ollama
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.chunks = []
        self.chunk_embeddings = []
        self.entities = {}
        self.concepts = {}
        self.article_titles = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # Verificar que Ollama esté corriendo
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Verifica que Ollama esté corriendo."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama conectado en {self.ollama_url}")
                # Listar modelos disponibles
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"  Modelos disponibles: {', '.join(model_names[:5])}")

                # Verificar que el modelo específico esté disponible
                if self.model_name not in model_names:
                    print(f"\n⚠️  Modelo '{self.model_name}' no encontrado.")
                    print(f"   Modelos disponibles: {', '.join(model_names)}")
                    print(f"\n   Para descargar el modelo, ejecuta:")
                    print(f"   ollama pull {self.model_name}")
            else:
                print(f"ERROR: Ollama no responde en {self.ollama_url}")
        except requests.exceptions.ConnectionError:
            print(f"ERROR: No se pudo conectar a Ollama en {self.ollama_url}")
            print("Asegúrate de que Ollama esté corriendo:")
            print("  1. Abre Ollama desde el menú de inicio")
            print("  2. O ejecuta 'ollama serve' en la terminal")
        except Exception as e:
            print(f"ERROR verificando Ollama: {e}")

    def load_data(self, chunks_path: str, embeddings_path: str, entities_path: str):
        """
        Carga los datos procesados (chunks, embeddings, entidades).

        Args:
            chunks_path: Ruta al CSV de chunks
            embeddings_path: Ruta al JSON de embeddings
            entities_path: Ruta al JSON de entidades
        """
        # Cargar chunks
        import pandas as pd
        chunks_df = pd.read_csv(chunks_path)
        self.chunks = chunks_df['chunk_text'].tolist()
        self.article_titles = chunks_df['article_title'].tolist()

        # Limpiar chunks para eliminar referencias, citas y ruido
        print("Limpiando chunks...")
        self.chunks = [self._clean_chunk_text(chunk) for chunk in self.chunks]

        # Cargar embeddings
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
            self.chunk_embeddings = [np.array(e['embedding']) for e in embeddings_data]

        # Cargar entidades y conceptos
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
            self.entities = entities_data.get('entities', {})
            self.concepts = entities_data.get('concepts', {})

        print(f"✓ Datos cargados: {len(self.chunks)} chunks, {len(self.chunk_embeddings)} embeddings")
        print(f"✓ Artículos: {len(set(self.article_titles))}")
        print(f"✓ Entidades: {len(self.entities)} artículos con entidades")

        # Crear matriz TF-IDF para recuperación más efectiva
        print("Creando matriz TF-IDF para recuperación...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Unigramas y bigramas
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        print(f"✓ Matriz TF-IDF creada: {self.tfidf_matrix.shape}")

    def _clean_chunk_text(self, text: str) -> str:
        """
        Limpia el texto de un chunk eliminando referencias y secciones de ruido.

        Args:
            text: Texto del chunk

        Returns:
            Texto limpio
        """
        import re

        # Cortar secciones de referencias/bibliografía que suelen ir al final
        text = re.split(
            r'(?:^|\s)(?:References|External links|See also|Notes|Bibliography|'
            r'Further reading|Sources|Works cited|Footnotes)\s*(?:\[|\(|:|$)',
            text, flags=re.IGNORECASE
        )[0]

        # Eliminar referencias [1], [2], etc. y links markdown
        cleaned = re.sub(r'\[\d+\]', '', text)
        cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Eliminar oraciones que son puras citas bibliográficas
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        sentences = [
            s for s in sentences
            if not re.search(
                r'archived from the original|retrieved |doi\.org|isbn|'
                r'pp\.\s*\d|vol\.\s*\d|no\.\s*\d{1,2}\b', s, re.IGNORECASE
            )
        ]
        cleaned = ' '.join(sentences)

        return cleaned.strip()

    def _extract_relevant_sentences(self, chunk: str, query: str, max_chars: int = 1000) -> str:
        """
        Extrae las oraciones de un chunk más relevantes para la consulta.

        Args:
            chunk: Texto del chunk
            query: Consulta del usuario
            max_chars: Máximo de caracteres a retornar

        Returns:
            Oraciones relevantes unidas
        """
        import re

        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        if not sentences:
            return chunk[:max_chars]

        # Palabras de contenido de la consulta
        query_words = set(re.findall(r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,}', query.lower()))
        stop = {
            'que', 'para', 'como', 'con', 'una', 'los', 'las', 'del', 'por',
            'the', 'and', 'for', 'are', 'was', 'were', 'what', 'who', 'how',
            'why', 'does', 'did', 'you', 'your', 'this', 'that', 'from', 'have',
            'has', 'had', 'not', 'but', 'all', 'can', 'her', 'his', 'its', 'our',
            'their', 'they', 'them', 'when', 'where', 'which', 'about', 'into',
            'quien', 'quienes', 'cual', 'cuales', 'donde', 'cuando', 'porque',
            'pero', 'este', 'esta', 'estos', 'estas', 'eso', 'esa', 'esos',
            'esas', 'fue', 'son', 'ser', 'han', 'hay', 'muy', 'mas', 'tambien',
            'fue', 'era', 'eran', 'sobre', 'entre', 'hasta', 'desde', 'durante'
        }
        query_words -= stop

        if not query_words:
            # Sin palabras útiles: tomar el inicio del chunk (suele ser el lead)
            return chunk[:max_chars]

        scored = []
        for s in sentences:
            s_words = set(re.findall(r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,}', s.lower()))
            score = len(query_words & s_words)
            # Bonus si la oración es corta y densa en matches
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])

        out, total = [], 0
        for score, s in scored:
            if score == 0:
                break
            if total + len(s) > max_chars:
                continue
            out.append(s)
            total += len(s)

        if not out:
            # Ninguna oración contiene términos de la consulta: usar el inicio
            return chunk[:max_chars]

        return ' '.join(out)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding para una consulta usando spaCy (mismo que los chunks).

        Args:
            text: Texto a vectorizar

        Returns:
            Vector numpy (96 dimensiones para coincidir con chunks)
        """
        try:
            import spacy
            if not hasattr(self, 'nlp'):
                self.nlp = spacy.load("en_core_web_sm")
            doc = self.nlp(text)
            vectors = [token.vector for token in doc if token.has_vector]
            if not vectors:
                return np.zeros(96)
            return np.mean(vectors, axis=0)
        except Exception as e:
            print(f"ERROR generando embedding: {e}")
            return np.zeros(96)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str, str]]:
        """
        Recupera los chunks más relevantes para una consulta usando TF-IDF.

        Args:
            query: Consulta del usuario
            top_k: Número de chunks a retornar

        Returns:
            Lista de tuplas (índice, similitud, chunk, título del artículo)
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("TF-IDF no inicializado. Llama a load_data primero.")

        # Transformar la consulta
        query_tfidf = self.tfidf_vectorizer.transform([query])

        # Calcular similitud coseno entre consulta y todos los chunks
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # Obtener índices de los chunks más similares
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.chunks[idx],
                self.article_titles[idx]
            ))

        return results

    def generate_summary_ollama(self, text: str, prompt: str = "Summarize the following text in Spanish:") -> str:
        """
        Genera un resumen usando Ollama.

        Args:
            text: Texto a resumir (puede estar vacío si solo se usa prompt)
            prompt: Prompt para el modelo

        Returns:
            Resumen generado
        """
        try:
            # Preparar prompt
            full_prompt = prompt
            if text:
                full_prompt = f"{prompt}\n\n{text}"

            # Llamar a la API de Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': self.model_name,
                    'prompt': full_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 500  # Máximo de tokens a generar
                    }
                },
                timeout=120  # Timeout de 2 minutos
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"ERROR: Ollama retornó código {response.status_code}")
                return "No se pudo generar la respuesta."

        except requests.exceptions.Timeout:
            print("ERROR: Timeout al generar respuesta con Ollama")
            return "La respuesta tardó demasiado. Intenta de nuevo."
        except Exception as e:
            print(f"ERROR generando resumen con Ollama: {e}")
            return "Ocurrió un error al generar la respuesta."

    def answer_question_ollama(self, query: str, relevant_chunks: List[Tuple[int, float, str, str]], top_k: int = 3) -> str:
        """
        Responde una pregunta usando Ollama basándose en el contenido de los chunks.

        Args:
            query: Pregunta del usuario
            relevant_chunks: Lista de chunks relevantes
            top_k: Número de chunks a usar

        Returns:
            Respuesta generada
        """
        # Usar solo las oraciones relevantes de cada chunk (contexto más limpio y corto)
        context = '\n\n'.join([
            self._extract_relevant_sentences(chunk[2], query, max_chars=900)
            for chunk in relevant_chunks[:top_k]
        ])

        # Crear prompt para responder la pregunta
        prompt = f"""Eres un asistente que responde preguntas basándose en la siguiente información de Wikipedia.

INFORMACIÓN:
{context}

PREGUNTA: {query}

INSTRUCCIONES:
1. Responde la pregunta en español de forma clara y concisa
2. Usa SOLO la información proporcionada arriba
3. Si la información no es suficiente para responder, di "No tengo suficiente información para responder esta pregunta"
4. No inventes información que no esté en el contexto
5. Sé específico y directo

RESPUESTA:"""

        return self.generate_summary_ollama("", prompt)

    def extract_answer_from_chunks(self, query: str, chunks: List[Tuple[int, float, str, str]], top_k: int = 3) -> str:
        """
        Extrae una respuesta de los chunks relevantes.

        Args:
            query: Consulta del usuario
            chunks: Lista de chunks relevantes
            top_k: Número de chunks a usar para la respuesta

        Returns:
            Respuesta generada
        """
        return self.answer_question_ollama(query, chunks, top_k)

    def extract_related_concepts(self, chunks: List[Tuple[int, float, str, str]], top_k: int = 5) -> List[str]:
        """
        Extrae conceptos relacionados de los chunks relevantes.

        Args:
            chunks: Lista de chunks relevantes
            top_k: Número de conceptos a retornar

        Returns:
            Lista de conceptos relacionados
        """
        # Obtener conceptos de los artículos de los chunks relevantes
        concepts = []
        for _, _, _, article_title in chunks:
            if article_title in self.concepts:
                article_concepts = self.concepts[article_title]
                concepts.extend(article_concepts[:5])  # Top 5 por artículo

        # Eliminar duplicados manteniendo orden
        concepts = list(dict.fromkeys(concepts))
        return concepts[:top_k]

    def answer_question(self, query: str, top_k_chunks: int = 10) -> Dict:
        """
        Responde una pregunta del usuario.

        Args:
            query: Pregunta del usuario
            top_k_chunks: Número de chunks a considerar

        Returns:
            Diccionario con la respuesta, chunks relevantes y conceptos
        """
        # Paso 1: Recuperar chunks relevantes
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k_chunks)

        # Si la mejor similitud es muy baja, no hay contexto relevante del corpus.
        # En ese caso el modelo responde con su conocimiento general.
        if not relevant_chunks or relevant_chunks[0][1] < 0.12:
            answer = self.answer_general_question(query)
            return {
                'answer': answer,
                'chunks': [],
                'concepts': [],
                'source': 'general'
            }

        # Paso 2: Generar respuesta con Ollama (contexto acotado y relevante)
        answer = self.extract_answer_from_chunks(query, relevant_chunks, top_k=6)

        # Paso 3: Extraer conceptos relacionados
        concepts = self.extract_related_concepts(relevant_chunks, top_k=5)

        # Paso 4: Obtener entidades del chunk más relevante
        top_chunk = relevant_chunks[0]
        entities = {}
        if top_chunk[3] in self.entities:
            entities = self.entities[top_chunk[3]]

        return {
            'answer': answer,
            'chunks': [
                {
                    'index': idx,
                    'similarity': sim,
                    'text': chunk[:500] + '...' if len(chunk) > 500 else chunk,
                    'article': title
                }
                for idx, sim, chunk, title in relevant_chunks
            ],
            'concepts': concepts,
            'entities': entities,
            'top_similarity': relevant_chunks[0][1],
            'source': 'corpus'
        }

    def answer_general_question(self, query: str) -> str:
        """
        Responde una pregunta usando solo el conocimiento general del modelo,
        sin contexto de los documentos procesados.

        Args:
            query: Pregunta del usuario

        Returns:
            Respuesta generada
        """
        prompt = f"""Eres un asistente útil y conciso.

PREGUNTA: {query}

INSTRUCCIONES:
1. Responde en el mismo idioma en que se hizo la pregunta
2. Usa tus conocimientos generales
3. Sé claro, preciso y directo
4. Si no sabes la respuesta, dilo honestamente

RESPUESTA:"""

        return self.generate_summary_ollama("", prompt)

    def interactive_mode(self):
        """Modo interactivo para hacer preguntas."""
        print("\n" + "="*60)
        print("EXPLORADOR DE CONOCIMIENTO - MODO INTERACTIVO")
        print("="*60)
        print(f"Modelo: {self.model_name}")
        print("Escribe tu pregunta en español o 'exit' para salir\n")

        while True:
            query = input("> ").strip()

            if query.lower() in ['exit', 'quit', 'salir']:
                print("¡Hasta luego!")
                break

            if not query:
                continue

            print("\nBuscando información...")
            result = self.answer_question(query)

            print(f"\n{'='*60}")
            print("RESPUESTA:")
            print(f"{'='*60}")
            print(result['answer'])

            if result['concepts']:
                print(f"\n{'='*60}")
                print("CONCEPTOS RELACIONADOS:")
                print(f"{'='*60}")
                for concept in result['concepts']:
                    print(f"  • {concept}")

            if result['entities']:
                print(f"\n{'='*60}")
                print("ENTIDADES ENCONTRADAS:")
                print(f"{'='*60}")
                for entity_type, entity_list in result['entities'].items():
                    if entity_list:
                        print(f"  {entity_type}: {', '.join(entity_list[:3])}")

            if result['chunks']:
                print(f"\n{'='*60}")
                print("CHUNKS MÁS RELEVANTES:")
                print(f"{'='*60}")
                for chunk in result['chunks'][:3]:
                    print(f"\n  Artículo: {chunk['article']}")
                    print(f"  Similitud: {chunk['similarity']:.3f}")
                    print(f"  Texto: {chunk['text'][:200]}...")

            print(f"\n{'='*60}\n")


def main():
    """Función principal para probar el generador."""
    # Rutas de archivos
    base_dir = Path(__file__).parent.parent.parent
    chunks_path = base_dir / "data" / "processed" / "chunks_dataframe.csv"
    embeddings_path = base_dir / "data" / "processed" / "chunks_embeddings.json"
    entities_path = base_dir / "data" / "processed" / "entities.json"

    # Verificar que existan los archivos
    if not all(p.exists() for p in [chunks_path, embeddings_path, entities_path]):
        print("ERROR: No se encontraron los archivos procesados.")
        print("Primero ejecuta el pipeline de conocimiento.")
        return

    # Inicializar generador con Ollama
    generator = OllamaGenerator(model_name="qwen3:4b-instruct-2507-q4_K_M")
    generator.load_data(str(chunks_path), str(embeddings_path), str(entities_path))

    # Modo interactivo
    generator.interactive_mode()


if __name__ == "__main__":
    main()
