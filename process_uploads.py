"""
Procesa los documentos subidos en data/uploads/ y los fusiona con el corpus
procesado existente (o crea el corpus si no existe ninguno).

Se usa cuando el usuario sube archivos sin ingresar un tema de Wikipedia.
Los documentos se re-derivan por completo en cada ejecución: los artículos
marcados como 'uploaded_document' se reemplazan por la versión actual de
data/uploads/.
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd

# Configurar encoding UTF-8 para Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Rutas de módulos
base_dir = Path(__file__).parent
sys.path.append(str(base_dir / "src" / "data_extraction"))
sys.path.append(str(base_dir / "src" / "transformation"))
sys.path.append(str(base_dir / "src" / "embeddings"))

from document_loader import load_uploaded_documents
from data_transformation import DataTransformer
from semantic_analyzer import SemanticAnalyzer


def _is_upload(article: dict) -> bool:
    return 'uploaded_document' in (article.get('categories') or [])


def main():
    uploads_dir = base_dir / "data" / "uploads"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    transformed_path = processed_dir / "transformed_articles.json"
    chunks_path = processed_dir / "chunks_dataframe.csv"
    embeddings_path = processed_dir / "chunks_embeddings.json"
    entities_path = processed_dir / "entities.json"

    # 1. Cargar documentos subidos (divididos por tema)
    print("=" * 60)
    print("PROCESANDO DOCUMENTOS SUBIDOS")
    print("=" * 60)
    uploaded_articles = load_uploaded_documents(uploads_dir)

    if not uploaded_articles:
        print("No hay documentos para procesar en data/uploads/")
        return

    # 2. Cargar corpus existente, separando uploads viejos de artículos wiki
    existing_transformed = []
    if transformed_path.exists():
        with open(transformed_path, 'r', encoding='utf-8') as f:
            existing_transformed = json.load(f)
        print(f"Corpus existente: {len(existing_transformed)} artículos")

    wiki_articles = [a for a in existing_transformed if not _is_upload(a)]
    old_upload_titles = {a['title'] for a in existing_transformed if _is_upload(a)}
    wiki_titles = {a['title'] for a in wiki_articles}

    if old_upload_titles:
        print(f"Reemplazando {len(old_upload_titles)} artículo(s) de uploads anteriores")

    # 3. Transformar los documentos subidos (los uploads se re-derivan siempre)
    transformer = DataTransformer(str(base_dir / "data" / "raw" / "main_article.json"))
    analyzer = SemanticAnalyzer()

    new_transformed = []
    for article in uploaded_articles:
        if article['title'] in wiki_titles:
            print(f"  Colisión con artículo existente, se omite: {article['title']}")
            continue
        print(f"Transformando: {article['title']}")
        transformed = transformer.transform_article(article)
        new_transformed.append(transformed)

    if not new_transformed:
        print("No hay documentos nuevos que procesar.")
        return

    # 4. Fusionar: wiki existentes + uploads actuales
    all_transformed = wiki_articles + new_transformed

    with open(transformed_path, 'w', encoding='utf-8') as f:
        json.dump(all_transformed, f, ensure_ascii=False, indent=2)

    # Reconstruir DataFrame de chunks desde todos los artículos
    chunks_data = []
    for article in all_transformed:
        for i, chunk in enumerate(article['chunks']):
            chunks_data.append({
                'article_title': article['title'],
                'article_id': article['page_id'],
                'chunk_id': i,
                'chunk_text': chunk,
                'chunk_length': len(chunk.split())
            })
    chunks_df = pd.DataFrame(chunks_data)
    chunks_df.to_csv(chunks_path, index=False, encoding='utf-8')
    print(f"Chunks totales: {len(chunks_df)}")

    # 5. Embeddings y NER: purgar uploads viejos, agregar los nuevos
    existing_embeddings = []
    if embeddings_path.exists():
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            existing_embeddings = json.load(f)
    existing_embeddings = [
        e for e in existing_embeddings
        if e.get('article_title') not in old_upload_titles
    ]

    existing_entities = {}
    existing_concepts = {}
    if entities_path.exists():
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
            existing_entities = entities_data.get('entities', {})
            existing_concepts = entities_data.get('concepts', {})
    for t in old_upload_titles:
        existing_entities.pop(t, None)
        existing_concepts.pop(t, None)

    for article in new_transformed:
        title = article['title']
        print(f"\nGenerando embeddings/NER: {title}")

        for i, chunk in enumerate(article['chunks']):
            embedding = analyzer.get_chunk_embedding(chunk)
            existing_embeddings.append({
                'chunk_id': f"{title}_{i}",
                'chunk_text': chunk,
                'embedding': embedding.tolist(),
                'article_title': title,
                'article_id': article['page_id']
            })

        full_text = ' '.join(article['chunks'])
        existing_entities[title] = analyzer.extract_entities(full_text)
        existing_concepts[title] = analyzer.extract_key_concepts(full_text)

    with open(embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(existing_embeddings, f, ensure_ascii=False, indent=2)

    with open(entities_path, 'w', encoding='utf-8') as f:
        json.dump({
            'entities': existing_entities,
            'concepts': existing_concepts
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("DOCUMENTOS PROCESADOS EXITOSAMENTE")
    print("=" * 60)
    print(f"Artículos de uploads: {len(new_transformed)}")
    print(f"Total en corpus: {len(all_transformed)} artículos")
    print(f"Total de chunks: {len(chunks_df)}")


if __name__ == "__main__":
    main()
