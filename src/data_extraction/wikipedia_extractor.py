import requests
import json
import time
import sys
import os
import re
from typing import List, Dict, Optional
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class WikipediaExtractor:
    def __init__(self, language: str = "en", use_parse_action: bool = False):
        """
        Inicializa el extractor.

        Args:
            language: Código de idioma (ej: 'en' para inglés)
            use_parse_action: Si True, usa action=parse para obtener contenido completo
                            (más lento pero puede retornar más contenido)
        """
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.use_parse_action = use_parse_action
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NLP-Knowledge-Explorer/1.0 (Educational Purpose)'
        })

        # Configurar retry strategy con backoff exponencial
        retry_strategy = Retry(
            total=5,  # máximo 5 reintentos
            backoff_factor=2,  # espera 2s, 4s, 8s, 16s, 32s entre reintentos
            status_forcelist=[429, 500, 502, 503, 504],  # códigos de error que retry
            allowed_methods=["GET"]  # solo reintentar peticiones GET
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def search_article_title(self, query: str, limit: int = 5) -> Optional[str]:
        """
        Busca el título correcto de un artículo usando la API de búsqueda de Wikipedia.

        Args:
            query: Consulta de búsqueda
            limit: Número máximo de resultados a retornar

        Returns:
            El título del primer resultado encontrado, o None si no hay resultados
        """
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srwhat': 'text'  # buscar en títulos y contenido
        }

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'query' in data and 'search' in data['query'] and data['query']['search']:
                # Retornar el título del primer resultado
                first_result = data['query']['search'][0]
                return first_result['title']

            return None

        except Exception as e:
            print(f"ERROR: Error buscando título para '{query}': {e}")
            return None
    
    def get_article_content_with_parse(self, title: str, try_search: bool = True) -> Optional[Dict]:
        """
        Obtiene el contenido de un artículo usando action=parse (alternativa para contenido completo).

        Args:
            title: Título del artículo
            try_search: Si True, intenta buscar el título correcto si no se encuentra

        Returns:
            Diccionario con el contenido del artículo o None si falla
        """
        params = {
            'action': 'parse',
            'format': 'json',
            'page': title,
            'prop': 'text|categories|links|revisions',
            'redirects': True,
            'rvprop': 'timestamp',
            'rvslots': 'main'
        }

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Verificar si hubo error
            if 'error' in data:
                # Artículo no encontrado, intentar buscar el título correcto
                if try_search:
                    print(f"  Artículo '{title}' no encontrado, buscando título correcto...")
                    correct_title = self.search_article_title(title)
                    if correct_title:
                        print(f"  Título encontrado: '{correct_title}'. Reintentando...")
                        return self.get_article_content_with_parse(correct_title, try_search=False)
                    else:
                        print(f"WARNING: Artículo no encontrado: {title}")
                        return None
                else:
                    print(f"WARNING: Artículo no encontrado: {title}")
                    return None

            # Extraer texto HTML y convertir a texto plano usando BeautifulSoup
            html_text = data['parse']['text']['*']

            # Usar BeautifulSoup para una limpieza más completa
            soup = BeautifulSoup(html_text, 'html.parser')

            # Eliminar scripts, styles y otros elementos no deseados
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg']):
                element.decompose()

            # Eliminar secciones de edición, referencias y elementos de navegación
            for element in soup.select('.mw-editsection, .reference, #toc, .navbox, .metadata, .ambox'):
                element.decompose()

            # Obtener texto plano
            plain_text = soup.get_text(separator=' ')

            # Limpiar espacios múltiples y caracteres especiales
            plain_text = re.sub(r'\s+', ' ', plain_text)
            plain_text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', plain_text)  # Zero-width characters
            plain_text = plain_text.strip()

            # Extraer categorías
            categories = [cat['*'] for cat in data['parse'].get('categories', [])]

            # Extraer links
            links = [link['*'] for link in data['parse'].get('links', [])]

            # Extraer revisión
            last_revision = ''
            if 'revisions' in data['parse']:
                last_revision = data['parse']['revisions'][0].get('timestamp', '')

            return {
                'title': data['parse'].get('title', title),
                'page_id': data['parse'].get('pageid', ''),
                'extract': plain_text,
                'categories': categories,
                'links': links,
                'last_revision': last_revision,
                'extracted_at': datetime.now().isoformat()
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"WARNING: Rate limit excedido para '{title}' (se reintentará automáticamente)")
            else:
                print(f"ERROR: Error HTTP extrayendo '{title}': {e}")
            return None
        except Exception as e:
            print(f"ERROR: Error extrayendo '{title}': {e}")
            return None

    def get_article_content(self, title: str, try_search: bool = True) -> Optional[Dict]:
        """
        Obtiene el contenido de un artículo de Wikipedia.

        Args:
            title: Título del artículo
            try_search: Si True, intenta buscar el título correcto si no se encuentra

        Returns:
            Diccionario con el contenido del artículo o None si falla
        """
        # Usar action=parse si está habilitado
        if self.use_parse_action:
            return self.get_article_content_with_parse(title, try_search)

        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts|categories|links|revisions',
            'explaintext': True,
            'titles': title,
            'exintro': False,
            'redirects': True,
            'exsectionformat': 'wiki',  # Obtener formato wiki completo
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
                # Artículo no encontrado, intentar buscar el título correcto
                if try_search:
                    print(f"  Artículo '{title}' no encontrado, buscando título correcto...")
                    correct_title = self.search_article_title(title)
                    if correct_title:
                        print(f"  Título encontrado: '{correct_title}'. Reintentando...")
                        return self.get_article_content(correct_title, try_search=False)
                    else:
                        print(f"WARNING: Artículo no encontrado: {title}")
                        return None
                else:
                    print(f"WARNING: Artículo no encontrado: {title}")
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

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"WARNING: Rate limit excedido para '{title}' (se reintentará automáticamente)")
            else:
                print(f"ERROR: Error HTTP extrayendo '{title}': {e}")
            return None
        except Exception as e:
            print(f"ERROR: Error extrayendo '{title}': {e}")
            return None
    
    def extract_multiple_articles(self, titles: List[str], delay: float = 2.0) -> List[Dict]:
        """
        Extrae múltiples artículos con un delay entre peticiones.

        Args:
            titles: Lista de títulos de artículos
            delay: Segundos de espera entre peticiones (aumentado a 2.0 por defecto)

        Returns:
            Lista de diccionarios con el contenido de los artículos
        """
        articles = []

        for i, title in enumerate(titles, 1):
            print(f"[{i}/{len(titles)}] Extrayendo: {title}")
            article = self.get_article_content(title)

            if article:
                articles.append(article)
                print(f"OK '{title}' extraído exitosamente")
            else:
                print(f"✗ '{title}' no pudo ser extraído")

            # Delay entre peticiones (excepto después de la última)
            if i < len(titles):
                time.sleep(delay)

        return articles

    def extract_article_links(self, title: str) -> List[str]:
        """
        Extrae los links internos de un artículo (títulos de otros artículos de Wikipedia).

        Args:
            title: Título del artículo

        Returns:
            Lista de títulos de artículos relacionados
        """
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'links',
            'titles': title,
            'pllimit': 500,  # máximo 500 links
            'plnamespace': 0  # solo artículos (no categorías, archivos, etc.)
        }

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page_id = next(iter(pages))

            if page_id == '-1':
                return []

            page = pages[page_id]
            links = [link['title'] for link in page.get('links', [])]

            return links

        except Exception as e:
            print(f"ERROR: Error extrayendo links de '{title}': {e}")
            return []

    def crawl_related_articles(self, seed_title: str, max_depth: int = 1, max_articles: int = 10, links_per_article: int = 5) -> List[Dict]:
        """
        Extrae artículos relacionados siguiendo links de Wikipedia.

        Args:
            seed_title: Título del artículo inicial (semilla)
            max_depth: Profundidad del crawling (1 = solo links directos, 2 = links de links, etc.)
            max_articles: Número máximo de artículos a extraer
            links_per_article: Número máximo de links a seguir por artículo

        Returns:
            Lista de artículos extraídos (incluye el artículo semilla)
        """
        articles = []
        visited = set()

        def crawl_recursive(title: str, depth: int):
            # Condiciones de parada
            if depth > max_depth:
                return
            if len(articles) >= max_articles:
                return
            if title in visited:
                return

            visited.add(title)
            print(f"  Crawling: {title} (depth {depth})")

            article = self.get_article_content(title)

            if article:
                articles.append(article)
                print(f"    OK '{title}' extraído exitosamente")

                # Obtener links y crawlear si no hemos llegado al límite
                if len(articles) < max_articles:
                    links = self.extract_article_links(title)
                    print(f"    Encontrados {len(links)} links en '{title}'")

                    # Tomar solo los primeros N links para evitar explosión
                    links_to_follow = links[:links_per_article]

                    for link in links_to_follow:
                        if link not in visited and len(articles) < max_articles:
                            # Delay entre peticiones
                            time.sleep(2.0)
                            crawl_recursive(link, depth + 1)
            else:
                print(f"    ✗ '{title}' no pudo ser extraído")

        crawl_recursive(seed_title, 0)
        return articles

    def get_related_articles(self, title: str, limit: int = 10, exclude_lists: bool = True) -> List[str]:
        """
        Obtiene artículos relacionados usando el operador 'morelike:' de Wikipedia.
        Este método usa el algoritmo CirrusSearch de Wikipedia que considera
        frecuencia de palabras, enlaces y similitud de temas.

        Args:
            title: Título del artículo (ej: "Xbox")
            limit: Número máximo de artículos relacionados a retornar
            exclude_lists: Si True, excluye artículos que empiezan con "List of"

        Returns:
            Lista de títulos de artículos relacionados
        """
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': f'morelike:{title}',
            'srlimit': limit * 2 if exclude_lists else limit,  # Pedir más si vamos a filtrar
            'srwhat': 'text'
        }

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'query' in data and 'search' in data['query']:
                results = data['query']['search']

                # Filtrar: excluir el artículo original y opcionalmente listas
                related = []
                for r in results:
                    title_candidate = r['title']

                    # Excluir el artículo original
                    if title_candidate == title:
                        continue

                    # Excluir artículos "List of..." si está habilitado
                    if exclude_lists and title_candidate.lower().startswith(('list of', 'lists of')):
                        continue

                    related.append(title_candidate)

                    # Si ya tenemos suficientes, parar
                    if len(related) >= limit:
                        break

                return related[:limit]

            return []

        except Exception as e:
            print(f"ERROR: Error obteniendo artículos relacionados para '{title}': {e}")
            return []

    def crawl_related_articles_filtered(self, seed_title: str, max_depth: int = 1, max_articles: int = 10, links_per_article: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Extrae artículos relacionados siguiendo links, filtrando por similitud semántica.

        Args:
            seed_title: Título del artículo inicial (semilla)
            max_depth: Profundidad del crawling
            max_articles: Número máximo de artículos a extraer
            links_per_article: Número máximo de links a seguir por artículo
            similarity_threshold: Umbral mínimo de similitud (0-1) para considerar un link relevante

        Returns:
            Lista de artículos extraídos filtrados por relevancia
        """
        # Importar SemanticAnalyzer
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))
        from semantic_analyzer import SemanticAnalyzer

        analyzer = SemanticAnalyzer()
        articles = []
        visited = set()

        def crawl_recursive(title: str, depth: int):
            if depth > max_depth:
                return
            if len(articles) >= max_articles:
                return
            if title in visited:
                return

            visited.add(title)
            print(f"  Crawling: {title} (depth {depth})")

            article = self.get_article_content(title)

            if article:
                articles.append(article)
                print(f"    OK '{title}' extraído exitosamente")

                # Obtener links y filtrarlos por similitud
                if len(articles) < max_articles:
                    links = self.extract_article_links(title)
                    print(f"    Encontrados {len(links)} links en '{title}'")

                    # Calcular similitud de cada link con el título semilla
                    link_similarities = []
                    for link in links:
                        if link not in visited:
                            similarity = analyzer.get_title_similarity(seed_title, link)
                            link_similarities.append((link, similarity))

                    # Ordenar por similitud descendente
                    link_similarities.sort(key=lambda x: x[1], reverse=True)

                    # Filtrar por umbral y tomar los primeros N
                    relevant_links = [
                        (link, sim) for link, sim in link_similarities
                        if sim >= similarity_threshold
                    ][:links_per_article]

                    print(f"    Links relevantes (sim >= {similarity_threshold}): {len(relevant_links)}")

                    for link, similarity in relevant_links:
                        print(f"      -> {link} (sim: {similarity:.3f})")
                        if link not in visited and len(articles) < max_articles:
                            time.sleep(2.0)
                            crawl_recursive(link, depth + 1)
            else:
                print(f"    ✗ '{title}' no pudo ser extraído")

        crawl_recursive(seed_title, 0)
        return articles

    def crawl_related_articles_moredlike(self, seed_title: str, max_depth: int = 1, max_articles: int = 10, articles_per_level: int = 10, exclude_lists: bool = True) -> List[Dict]:
        """
        Extrae artículos relacionados usando el operador 'morelike:' de Wikipedia.
        Este método usa el algoritmo nativo de Wikipedia (CirrusSearch) para encontrar
        artículos relacionados, que es más preciso que los embeddings de spaCy.

        Args:
            seed_title: Título del artículo inicial (semilla)
            max_depth: Profundidad del crawling (1 = solo relacionados directos)
            max_articles: Número máximo de artículos a extraer
            articles_per_level: Número de artículos relacionados a obtener por nivel
            exclude_lists: Si True, excluye artículos "List of..."

        Returns:
            Lista de artículos extraídos (sin incluir el artículo semilla)
        """
        articles = []
        visited = set()
        queue = [(seed_title, 0)]  # (título, profundidad)

        while queue and len(articles) < max_articles:
            title, depth = queue.pop(0)  # FIFO (breadth-first)

            if title in visited or depth > max_depth:
                continue

            visited.add(title)

            # No procesar el artículo semilla (ya lo tenemos)
            if title == seed_title and depth == 0:
                # Obtener artículos relacionados del semilla
                related = self.get_related_articles(title, limit=articles_per_level, exclude_lists=exclude_lists)
                print(f"  Artículo semilla: {title}")
                print(f"    Artículos relacionados (morelike:): {len(related)}")

                for related_title in related:
                    if related_title not in visited and len(articles) + len(queue) < max_articles:
                        print(f"      -> {related_title}")
                        queue.append((related_title, depth + 1))
                continue

            print(f"  Crawling: {title} (depth {depth})")

            article = self.get_article_content(title)

            if article:
                articles.append(article)
                print(f"    OK '{title}' extraído exitosamente")

                # Obtener artículos relacionados usando morelike:
                if len(articles) < max_articles and depth < max_depth:
                    related = self.get_related_articles(title, limit=articles_per_level, exclude_lists=exclude_lists)
                    print(f"    Artículos relacionados (morelike:): {len(related)}")

                    for related_title in related:
                        if related_title not in visited and len(articles) + len(queue) < max_articles:
                            print(f"      -> {related_title}")
                            queue.append((related_title, depth + 1))
            else:
                print(f"    ✗ '{title}' no pudo ser extraído")

            # Delay entre peticiones
            if queue:
                time.sleep(2.0)

        return articles

    def save_to_json(self, articles: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Guardados {len(articles)} artículos en: {output_path}")


def main():
    print("=" * 60)
    print("EXTRACTOR DE ARTÍCULOS DE WIKIPEDIA")
    print("=" * 60)
    print("\nOpciones:")
    print("1. Extraer artículos predefinidos de NLP/ML")
    print("2. Crawlear artículos relacionados desde un tema (usando 'morelike:' de Wikipedia)")
    print("3. Salir")
    print()

    choice = input("Selecciona una opción (1-3): ").strip()

    if choice == "1":
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
        extractor = WikipediaExtractor(language="en", use_parse_action=True)

        print("\nIniciando extracción de artículos de Wikipedia...")
        print("Configuración: Backoff exponencial con retry (máx 5 reintentos)")
        print("Búsqueda de títulos: Habilitada (para artículos no encontrados)")
        print("Delay entre peticiones: 2 segundos\n")
        articles = extractor.extract_multiple_articles(articles_to_extract, delay=2.0)

    elif choice == "2":
        # Crawling usando morelike: de Wikipedia (RECOMENDADO)
        print("\nIngresa el tema/tema inicial para crawlear:")
        print("(Ejemplo: 'Xbox', 'Machine Learning', 'Natural Language Processing', 'The Legend of Zelda')")
        print("\nNOTA: Este método usa el algoritmo nativo de Wikipedia (CirrusSearch)")
        print("para encontrar artículos relacionados.")
        seed_title = input("Tema: ").strip()

        if not seed_title:
            print("ERROR: Debes ingresar un tema válido.")
            return

        print("\nConfiguración de crawling:")
        max_articles = input("Número máximo de artículos (default 10): ").strip()
        max_articles = int(max_articles) if max_articles.isdigit() else 10

        max_depth = input("Profundidad de crawling (default 1): ").strip()
        max_depth = int(max_depth) if max_depth.isdigit() else 1

        articles_per_level = input("Artículos relacionados por nivel (default 5): ").strip()
        articles_per_level = int(articles_per_level) if articles_per_level.isdigit() else 5

        extractor = WikipediaExtractor(language="en", use_parse_action=True)

        print(f"\nIniciando crawling desde '{seed_title}'...")
        print(f"Configuración: Máximo {max_articles} artículos, profundidad {max_depth}, {articles_per_level} relacionados por nivel")
        print("Delay entre peticiones: 2 segundos\n")

        articles = extractor.crawl_related_articles_moredlike(
            seed_title=seed_title,
            max_depth=max_depth,
            max_articles=max_articles,
            articles_per_level=articles_per_level
        )

    else:
        print("Saliendo...")
        return

    # Guardar resultados
    output_path = "data/raw/wikipedia_articles.json"
    extractor.save_to_json(articles, output_path)

    print(f"\nResumen:")
    print(f"   - Artículos extraídos: {len(articles)}")
    print(f"   - Tasa de éxito: {len(articles)}/20 = {len(articles)/20*100:.1f}%" if choice == "1" else f"   - Artículos extraídos: {len(articles)}")


if __name__ == "__main__":
    main()