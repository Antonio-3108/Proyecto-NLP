"""
Pipeline principal del proyecto NLP.
Flujo: Extracción artículo principal → Transformación → Embeddings/NER →
       Extracción relacionados → Transformación → Embeddings/NER final
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Configurar encoding UTF-8 para Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Agregar directorios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_extraction'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'transformation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'embeddings'))

from wikipedia_extractor import WikipediaExtractor
from data_transformation import DataTransformer
from semantic_analyzer import SemanticAnalyzer
from document_loader import load_uploaded_documents


class KnowledgePipeline:
    """Pipeline para construir base de conocimiento desde Wikipedia."""

    def __init__(self):
        self.extractor = WikipediaExtractor(language="en", use_parse_action=True)
        self.analyzer = SemanticAnalyzer()
        self.base_dir = Path(__file__).parent

        # Directorios de datos
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Archivos de salida
        self.main_article_path = self.raw_dir / "main_article.json"
        self.related_articles_path = self.raw_dir / "related_articles.json"
        self.transformed_path = self.processed_dir / "transformed_articles.json"
        self.chunks_path = self.processed_dir / "chunks_dataframe.csv"
        self.embeddings_path = self.processed_dir / "chunks_embeddings.json"
        self.entities_path = self.processed_dir / "entities.json"

    def step1_extract_main_article(self, topic: str) -> Dict:
        """
        Paso 1: Extraer el artículo principal del tema.

        Args:
            topic: Tema/artículo inicial

        Returns:
            Artículo extraído
        """
        print("\n" + "="*60)
        print("PASO 1: Extracción del artículo principal")
        print("="*60)

        print(f"Extrayendo artículo principal: '{topic}'...")
        article = self.extractor.get_article_content(topic)

        if not article:
            print(f"ERROR: No se pudo extraer el artículo '{topic}'")
            return None

        print(f"OK Artículo extraído: {article['title']}")
        print(f"  - Longitud: {len(article['extract'])} caracteres")
        print(f"  - Categorías: {len(article['categories'])}")
        print(f"  - Links: {len(article['links'])}")

        # Guardar artículo principal
        with open(self.main_article_path, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        print(f"  Guardado en: {self.main_article_path}")

        return article

    def step2_transform_main_article(self, article: Dict) -> Dict:
        """
        Paso 2: Transformar el artículo principal.

        Args:
            article: Artículo a transformar

        Returns:
            Artículo transformado
        """
        print("\n" + "="*60)
        print("PASO 2: Transformación del artículo principal")
        print("="*60)

        transformer = DataTransformer(str(self.main_article_path))
        transformed = transformer.transform_article(article)

        print(f"OK Artículo transformado: {transformed['title']}")
        print(f"  - Chunks generados: {transformed['num_chunks']}")
        print(f"  - Longitud promedio de chunk: {sum(len(c.split()) for c in transformed['chunks'])/len(transformed['chunks']):.1f} palabras")

        return transformed

    def step3_embeddings_and_ner_main(self, transformed_article: Dict) -> Dict:
        """
        Paso 3: Generar embeddings y NER del artículo principal.

        Args:
            transformed_article: Artículo transformado

        Returns:
            Diccionario con embeddings y entidades
        """
        print("\n" + "="*60)
        print("PASO 3: Embeddings y NER del artículo principal")
        print("="*60)

        # Generar embeddings para cada chunk
        chunk_embeddings = []
        for i, chunk in enumerate(transformed_article['chunks']):
            embedding = self.analyzer.get_chunk_embedding(chunk)
            chunk_embeddings.append({
                'chunk_id': i,
                'chunk_text': chunk,
                'embedding': embedding.tolist(),
                'article_title': transformed_article['title']
            })
            print(f"  Chunk {i}: embedding generado ({len(embedding)} dimensiones)")

        # Extraer entidades del artículo completo
        full_text = ' '.join(transformed_article['chunks'])
        entities = self.analyzer.extract_entities(full_text)
        concepts = self.analyzer.extract_key_concepts(full_text)

        print(f"OK Embeddings generados: {len(chunk_embeddings)} chunks")
        print(f"OK Entidades extraídas: {len(entities)} tipos")
        print(f"OK Conceptos clave: {len(concepts)}")

        result = {
            'chunk_embeddings': chunk_embeddings,
            'entities': entities,
            'concepts': concepts,
            'article_title': transformed_article['title']
        }

        # Guardar embeddings y entidades del artículo principal
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_embeddings, f, ensure_ascii=False, indent=2)

        with open(self.entities_path, 'w', encoding='utf-8') as f:
            json.dump({'entities': entities, 'concepts': concepts}, f, ensure_ascii=False, indent=2)

        return result

    def step4_extract_related_articles(self, main_article: Dict, max_articles: int = 10, max_depth: int = 1) -> List[Dict]:
        """
        Paso 4: Extraer artículos relacionados usando morelike:.

        Args:
            main_article: Artículo principal
            max_articles: Número máximo de artículos a extraer
            max_depth: Profundidad del crawling

        Returns:
            Lista de artículos relacionados (sin incluir el artículo principal)
        """
        print("\n" + "="*60)
        print("PASO 4: Extracción de artículos relacionados")
        print("="*60)

        title = main_article['title']
        print(f"Buscando artículos relacionados para: '{title}'...")
        print(f"Filtrando artículos 'List of...' para mejor calidad")

        articles = self.extractor.crawl_related_articles_moredlike(
            seed_title=title,
            max_depth=max_depth,
            max_articles=max_articles,
            articles_per_level=10,  # Aumentado de 5 a 10 para más breadth
            exclude_lists=True  # Excluir "List of..." articles
        )

        # Eliminar duplicados del artículo principal por si acaso
        main_title = main_article['title']
        articles = [a for a in articles if a['title'] != main_title]

        print(f"OK Artículos relacionados extraídos: {len(articles)}")

        # Guardar artículos relacionados
        with open(self.related_articles_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"  Guardados en: {self.related_articles_path}")

        return articles

    def step5_transform_all_articles(self, main_article: Dict, related_articles: List[Dict]) -> List[Dict]:
        """
        Paso 5: Transformar todos los artículos (principal + relacionados).

        Args:
            main_article: Artículo principal
            related_articles: Lista de artículos relacionados

        Returns:
            Lista de artículos transformados
        """
        print("\n" + "="*60)
        print("PASO 5: Transformación de todos los artículos")
        print("="*60)

        # Documentos subidos por el usuario (data/uploads/) se procesan
        # igual que los artículos de Wikipedia
        uploads_dir = self.base_dir / "data" / "uploads"
        uploaded_articles = load_uploaded_documents(uploads_dir)
        if uploaded_articles:
            print(f"Documentos subidos incluidos: {len(uploaded_articles)}")

        # Combinar artículos y eliminar duplicados por título
        # Orden: principal -> uploads -> relacionados (uploads cerca del principal
        # para que tengan buen puntaje de relevancia en el grafo)
        all_articles = [main_article] + uploaded_articles + related_articles

        # Eliminar duplicados por título
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)

        all_articles = unique_articles
        print(f"Transformando {len(all_articles)} artículos (sin duplicados)...")

        transformer = DataTransformer(str(self.main_article_path))
        transformed_articles = []

        for i, article in enumerate(all_articles, 1):
            print(f"[{i}/{len(all_articles)}] Procesando: {article['title']}")
            transformed = transformer.transform_article(article)
            transformed_articles.append(transformed)

        # Guardar artículos transformados
        with open(self.transformed_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_articles, f, ensure_ascii=False, indent=2)
        print(f"OK Artículos transformados guardados en: {self.transformed_path}")

        # Crear DataFrame de chunks
        chunks_data = []
        for article in transformed_articles:
            for i, chunk in enumerate(article['chunks']):
                chunks_data.append({
                    'article_title': article['title'],
                    'article_id': article['page_id'],
                    'chunk_id': i,
                    'chunk_text': chunk,
                    'chunk_length': len(chunk.split())
                })

        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_csv(self.chunks_path, index=False, encoding='utf-8')
        print(f"OK DataFrame de chunks guardado en: {self.chunks_path}")
        print(f"  - Total de chunks: {len(chunks_df)}")
        print(f"  - Promedio de chunks por artículo: {len(chunks_df)/len(transformed_articles):.1f}")

        return transformed_articles

    def step6_embeddings_and_ner_final(self, transformed_articles: List[Dict]) -> Dict:
        """
        Paso 6: Generar embeddings y NER de todos los artículos.

        Args:
            transformed_articles: Lista de artículos transformados

        Returns:
            Diccionario con embeddings y entidades de todos los chunks
        """
        print("\n" + "="*60)
        print("PASO 6: Embeddings y NER de todos los artículos")
        print("="*60)

        all_chunk_embeddings = []
        all_entities = {}
        all_concepts = {}

        for article in transformed_articles:
            title = article['title']
            print(f"\nProcesando: {title}")

            # Generar embeddings para cada chunk
            for i, chunk in enumerate(article['chunks']):
                embedding = self.analyzer.get_chunk_embedding(chunk)
                all_chunk_embeddings.append({
                    'chunk_id': f"{title}_{i}",
                    'chunk_text': chunk,
                    'embedding': embedding.tolist(),
                    'article_title': title,
                    'article_id': article['page_id']
                })

            # Extraer entidades y conceptos del artículo completo
            full_text = ' '.join(article['chunks'])
            entities = self.analyzer.extract_entities(full_text)
            concepts = self.analyzer.extract_key_concepts(full_text)

            all_entities[title] = entities
            all_concepts[title] = concepts

            print(f"  - Chunks: {len(article['chunks'])}")
            print(f"  - Entidades: {len(entities)} tipos")
            print(f"  - Conceptos: {len(concepts)}")

        # Guardar embeddings y entidades finales
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunk_embeddings, f, ensure_ascii=False, indent=2)

        with open(self.entities_path, 'w', encoding='utf-8') as f:
            json.dump({
                'entities': all_entities,
                'concepts': all_concepts
            }, f, ensure_ascii=False, indent=2)

        print(f"\nOK Embeddings finales generados: {len(all_chunk_embeddings)} chunks")
        print(f"OK Entidades y conceptos guardados")

        return {
            'chunk_embeddings': all_chunk_embeddings,
            'entities': all_entities,
            'concepts': all_concepts
        }

    def run_pipeline(self, topic: str, max_related: int = 15, max_depth: int = 2):
        """
        Ejecuta el pipeline completo.

        Args:
            topic: Tema inicial
            max_related: Número máximo de artículos relacionados
            max_depth: Profundidad del crawling
        """
        print("\n" + "="*60)
        print("PIPELINE DE CONOCIMIENTO - INICIANDO")
        print("="*60)
        print(f"Tema: {topic}")
        print(f"Máximo de artículos relacionados: {max_related}")
        print(f"Profundidad de crawling: {max_depth}")

        # Paso 1: Extraer artículo principal
        main_article = self.step1_extract_main_article(topic)
        if not main_article:
            print("ERROR: No se pudo extraer el artículo principal. Abortando.")
            return

        # Paso 2: Transformar artículo principal
        transformed_main = self.step2_transform_main_article(main_article)

        # Paso 3: (Omitido - el paso 6 procesa todos los artículos incluyendo el principal)
        # main_analysis = self.step3_embeddings_and_ner_main(transformed_main)

        # Paso 4: Extraer artículos relacionados
        related_articles = self.step4_extract_related_articles(
            main_article,
            max_articles=max_related,
            max_depth=max_depth
        )

        # Paso 5: Transformar todos los artículos
        all_transformed = self.step5_transform_all_articles(main_article, related_articles)

        # Paso 6: Embeddings y NER final de todos los artículos
        final_analysis = self.step6_embeddings_and_ner_final(all_transformed)

        # Resumen final
        print("\n" + "="*60)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"\nResumen:")
        print(f"  - Artículo principal: {main_article['title']}")
        print(f"  - Artículos relacionados: {len(related_articles)}")
        print(f"  - Total de artículos: {len(all_transformed)}")
        print(f"  - Total de chunks: {len(final_analysis['chunk_embeddings'])}")
        print(f"  - Artículos con entidades: {len(final_analysis['entities'])}")

        print(f"\nArchivos generados:")
        print(f"  - {self.main_article_path}")
        print(f"  - {self.related_articles_path}")
        print(f"  - {self.transformed_path}")
        print(f"  - {self.chunks_path}")
        print(f"  - {self.embeddings_path}")
        print(f"  - {self.entities_path}")


def main():
    """Función principal."""
    pipeline = KnowledgePipeline()

    print("="*60)
    print("PIPELINE DE CONOCIMIENTO - EXPLORADOR INTERACTIVO")
    print("="*60)
    print("\nEste pipeline sigue el flujo:")
    print("1. Extracción del artículo principal")
    print("2. Transformación del artículo principal")
    print("3. Embeddings y NER del artículo principal")
    print("4. Extracción de artículos relacionados")
    print("5. Transformación de todos los artículos")
    print("6. Embeddings y NER final")
    print()

    # Ingresar tema
    topic = input("Ingresa el tema inicial (ej: 'Xbox', 'Machine Learning', 'Natural Language Processing'): ").strip()

    if not topic:
        print("ERROR: Debes ingresar un tema válido.")
        return

    # Configuración - usar valores por defecto si no hay input (modo servidor)
    try:
        max_related_input = input("Número máximo de artículos relacionados (default 10): ").strip()
        max_related = int(max_related_input) if max_related_input.isdigit() else 10
    except EOFError:
        # Modo servidor - usar valores por defecto
        max_related = 10

    try:
        max_depth_input = input("Profundidad de crawling (default 1): ").strip()
        max_depth = int(max_depth_input) if max_depth_input.isdigit() else 1
    except EOFError:
        # Modo servidor - usar valores por defecto
        max_depth = 1

    # Ejecutar pipeline
    pipeline.run_pipeline(topic, max_related=max_related, max_depth=max_depth)


if __name__ == "__main__":
    main()
