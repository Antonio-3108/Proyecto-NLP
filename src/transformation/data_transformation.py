import json
import pandas as pd
import re
import nltk
from pathlib import Path
from typing import List, Dict
import spacy


class DataTransformer:
    def __init__(self, data_path: str, output_dir: str = "data/processed"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Descargar recursos de NLTK
        self._download_nltk_resources()
        
        # Cargar modelo de spaCy (si está disponible)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            self.nlp = None
            self.use_spacy = False
    
    def _download_nltk_resources(self):
        required_resources = ['punkt', 'stopwords', 'wordnet']
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Descargando recurso NLTK: {resource} ...")
                nltk.download(resource, quiet=True)
    
    def clean_text(self, text: str) -> str:
        # Eliminar espacios en blanco múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres especiales no deseados (mantener puntuación básica)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Eliminar referencias numéricas de Wikipedia como [1], [23], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Limpiar espacios nuevamente
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        # Convertir a minúsculas
        text = text.lower()
        
        return text
    
    def remove_stopwords(self, text: str, use_spacy: bool = False) -> str:
        if use_spacy and self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_stop]
        else:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            tokens = text.split()
            tokens = [t for t in tokens if t not in stop_words]
        
        return ' '.join(tokens)
    
    def segment_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def transform_article(self, article: Dict) -> Dict:
        # Texto original
        original_text = article['extract']
        
        # Limpieza
        cleaned_text = self.clean_text(original_text)
        
        # Normalización
        normalized_text = self.normalize_text(cleaned_text)
        
        # Segmentación
        chunks = self.segment_text(cleaned_text, chunk_size=500, overlap=50)
        
        return {
            'title': article['title'],
            'page_id': article['page_id'],
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'normalized_text': normalized_text,
            'chunks': chunks,
            'num_chunks': len(chunks),
            'categories': article['categories'],
            'links': article['links'],
            'last_revision': article['last_revision'],
            'extracted_at': article['extracted_at']
        }
    
    def transform_corpus(self) -> List[Dict]:
        # Cargar datos
        with open(self.data_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Transformando {len(articles)} artículos...")
        
        transformed_articles = []
        for i, article in enumerate(articles, 1):
            print(f"[{i}/{len(articles)}] Procesando: {article['title']}")
            transformed = self.transform_article(article)
            transformed_articles.append(transformed)
        
        return transformed_articles
    
    def save_transformed_data(self, transformed_articles: List[Dict]):
        # Guardar JSON completo
        json_path = self.output_dir / "transformed_articles.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_articles, f, ensure_ascii=False, indent=2)
        print(f"Datos transformados guardados en: {json_path}")
        
        # Crear DataFrame de chunks para procesamiento posterior
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
        chunks_path = self.output_dir / "chunks_dataframe.csv"
        chunks_df.to_csv(chunks_path, index=False, encoding='utf-8')
        print(f"DataFrame de chunks guardado en: {chunks_path}")
        
        # Estadísticas
        print(f"\nEstadísticas de transformación:")
        print(f" - Total de artículos: {len(transformed_articles)}")
        print(f" - Total de chunks: {len(chunks_df)}")
        print(f" - Promedio de chunks por artículo: {len(chunks_df)/len(transformed_articles):.1f}")
        print(f" - Promedio de palabras por chunk: {chunks_df['chunk_length'].mean():.1f}")


def main():
    data_path = "data/raw/wikipedia_articles.json"
    
    if not Path(data_path).exists():
        print(f"No se encontró el archivo: {data_path}")
        print("Primero ejecuta el script de extracción de datos.")
        return
    
    transformer = DataTransformer(data_path)
    transformed_articles = transformer.transform_corpus()
    transformer.save_transformed_data(transformed_articles)
    
    print("\n" + "=" * 60)
    print("TRANSFORMACIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    main()