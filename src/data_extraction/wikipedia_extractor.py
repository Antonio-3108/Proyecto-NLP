import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime


class WikipediaExtractor:
    def __init__(self, language: str = "en"):
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NLP-Knowledge-Explorer/1.0 (Educational Purpose)'
        })
    
    def get_article_content(self, title: str) -> Optional[Dict]:
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts|categories|links|revisions',
            'explaintext': True,
            'titles': title,
            'exintro': False,
            'redirects': True,
            'cllimit': 500,
            'pllimit': 500,
            'rvprop': 'timestamp',
            'rvslots': 'main'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            page_id = next(iter(pages))
            
            if page_id == '-1':
                print(f"Artículo no encontrado: {title}")
                return None
            
            page = pages[page_id]
            
            return {
                'title': page.get('title', ''),
                'page_id': page_id,
                'extract': page.get('extract', ''),
                'categories': [cat['title'] for cat in page.get('categories', [])],
                'links': [link['title'] for link in page.get('links', [])],
                'last_revision': page.get('revisions', [{}])[0].get('timestamp', ''),
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extrayendo '{title}': {e}")
            return None
    
    def extract_multiple_articles(self, titles: List[str], delay: float = 1.0) -> List[Dict]:
        articles = []
        
        for i, title in enumerate(titles, 1):
            print(f"[{i}/{len(titles)}] Extrayendo: {title}")
            article = self.get_article_content(title)
            
            if article:
                articles.append(article)
            
            if i < len(titles):
                time.sleep(delay)
        
        return articles
    
    def save_to_json(self, articles: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Guardados {len(articles)} artículos en: {output_path}")


def main():
    # Lista de artículos relacionados con NLP y ML
    articles_to_extract = [
        "Natural language processing",
        "Tokenization",
        "TF–IDF",
        "Word2Vec",
        "Word embedding",
        "Named-entity recognition",
        "Recurrent neural network",
        "Long short-term memory",
        "Attention (machine learning)",
        "Transformer (machine learning model)",
        "BERT (language model)",
        "GPT (language model)",
        "Text classification",
        "Language model",
        "Word sense disambiguation",
        "Sentiment analysis",
        "Part-of-speech tagging",
        "Stop word",
        "N-gram",
        "Bag-of-words model"
    ]
    
    # Crear extractor
    extractor = WikipediaExtractor(language="en")
    
    # Extraer artículos
    print("Iniciando extracción de artículos de Wikipedia...")
    articles = extractor.extract_multiple_articles(articles_to_extract, delay=1.0)
    
    # Guardar resultados
    output_path = "data/raw/wikipedia_articles.json"
    extractor.save_to_json(articles, output_path)
    
    print(f"\nResumen:")
    print(f" - Artículos solicitados: {len(articles_to_extract)}")
    print(f" - Artículos extraídos: {len(articles)}")
    print(f" - Tasa de éxito: {len(articles)/len(articles_to_extract)*100:.1f}%")


if __name__ == "__main__":
    main()