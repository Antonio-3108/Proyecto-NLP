"""
Procesa los documentos subidos en data/uploads/ y los fusiona con el corpus
procesado existente (o crea el corpus si no existe ninguno).

Se usa cuando el usuario sube archivos sin ingresar un tema de Wikipedia.
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


def main():
    uploads_dir = base_dir / "data" / "uploads"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    transformed_path = processed_dir / "transformed_articles.json"
    chunks_path = processed_dir / "chunks_dataframe.csv"
    embeddings_path = processed_dir / "chunks_embeddings.json"
    entities_path = processed_dir / "entities.json"

    # 1. Cargar documentos subidos
    print("=" * 60)
    print("PROCESANDO DOCUMENTOS SUBIDOS")
    print("=" * 60)
    uploaded_articles = load_uploaded_documents(uploads_dir)

    if not uploaded_articles:
        print("No hay documentos para procesar en data/uploads/")
        return

    # 2. Cargar corpus existente (si hay)
    existing_transformed = []
    if transformed_path.exists():
        with open(transformed_path, 'r', encoding='utf-8') as f:
            existing_transformed = json.load(f)
        print(f"Corpus existente: {len(existing_transformed)} artículos")

    existing_titles = {a['title'] for a in existing_transformed}

    # 3. Transformar solo los documentos nuevos
    transformer = DataTransformer(str(base_dir / "data" / "raw" / "main_article.json"))
    analyzer = SemanticAnalyzer()

    new_transformed = []
    for article in uploaded_articles:
        if article['title'] in existing_titles:
            print(f"  Ya existe, se omite: {article['title']}")
            continue
        print(f"Transformando: {article['title']}")
        transformed = transformer.transform_article(article)
        new_transformed.append(transformed)

    if not new_transformed:
        print("No hay documentos nuevos que procesar.")
        return

    # 4. Fusionar con el corpus existente
    all_transformed = existing_transformed + new_transformed

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

    # 5. Embeddings y NER solo para los nuevos artículos
    existing_embeddings = []
    if embeddings_path.exists():
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            existing_embeddings = json.load(f)

    existing_entities = {}
    existing_concepts = {}
    if entities_path.exists():
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
            existing_entities = entities_data.get('entities', {})
            existing_concepts = entities_data.get('concepts', {})

    new_chunk_ids = set()
    for article in new_transformed:
        title = article['title']
        print(f"\nGenerando embeddings/NER: {title}")

        for i, chunk in enumerate(article['chunks']):
            embedding = analyzer.get_chunk_embedding(chunk)
            new_chunk_ids.add(f"{title}_{i}")
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
    print(f"Nuevos artículos: {len(new_transformed)}")
    print(f"Total en corpus: {len(all_transformed)} artículos")
    print(f"Total de chunks: {len(chunks_df)}")


if __name__ == "__main__":
    main()
