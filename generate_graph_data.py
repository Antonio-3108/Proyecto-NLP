"""
Script para generar datos del grafo en formato JSON para la aplicación web.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def classify_article_type(title: str, chunks: List[Dict]) -> str:
    """
    Clasifica un artículo por tipo de contenido basado en su título y contenido.

    Args:
        title: Título del artículo
        chunks: Lista de chunks del artículo

    Returns:
        Tipo de contenido: 'biography', 'company', 'product', 'technology', 'event', 'location', 'other'
    """
    title_lower = title.lower()

    # Biografías: nombres de personas, "born", "died", años de nacimiento
    biography_keywords = ['born', 'died', 'biography', 'life', 'career', 'president', 'ceo', 'director']
    if any(keyword in title_lower for keyword in biography_keywords):
        return 'biography'

    # Verificar contenido para más precisión
    if chunks:
        content_sample = ' '.join([chunk['text'][:200] for chunk in chunks[:2]]).lower()
        if any(keyword in content_sample for keyword in ['born', 'died', 'president', 'ceo', 'director', 'biography']):
            return 'biography'

    # Empresas: "company", "corporation", "inc", "ltd", "Inc."
    company_keywords = ['company', 'corporation', 'inc', 'ltd', 'inc.', 'ltd.', 'corp', 'group', 'holdings']
    if any(keyword in title_lower for keyword in company_keywords):
        return 'company'

    # Productos/Juegos: "game", "console", "software", "hardware", "app"
    product_keywords = ['game', 'console', 'software', 'hardware', 'app', 'system', 'platform', 'device']
    if any(keyword in title_lower for keyword in product_keywords):
        return 'product'

    # Tecnología: "technology", "computer", "internet", "software", "hardware"
    tech_keywords = ['technology', 'computer', 'internet', 'software', 'hardware', 'digital', 'ai', 'ml']
    if any(keyword in title_lower for keyword in tech_keywords):
        return 'technology'

    # Eventos: "event", "conference", "championship", "olympics", "world cup"
    event_keywords = ['event', 'conference', 'championship', 'olympics', 'world cup', 'tournament']
    if any(keyword in title_lower for keyword in event_keywords):
        return 'event'

    # Ubicaciones: países, ciudades, regiones
    location_keywords = ['united states', 'japan', 'europe', 'asia', 'africa', 'america', 'city', 'country']
    if any(keyword in title_lower for keyword in location_keywords):
        return 'location'

    # Verificar contenido para casos ambiguos
    if chunks:
        content_sample = ' '.join([chunk['text'][:300] for chunk in chunks[:2]]).lower()

        # Si menciona "video game" o "gaming" frecuentemente
        if 'video game' in content_sample or 'gaming' in content_sample:
            return 'product'

        # Si menciona "company" o "corporation" frecuentemente
        if 'company' in content_sample or 'corporation' in content_sample:
            return 'company'

    return 'other'


def extract_shared_concepts(article_texts: Dict[str, str], min_docs: int = 2,
                            max_concepts: int = 5) -> Dict[str, List[str]]:
    """
    Extrae frases (sustantivos compuestos) compartidas por varios artículos
    para usarlas como conceptos/nodos intermedios del grafo.

    Args:
        article_texts: {título: texto} de los artículos conservados
        min_docs: Mínimo de artículos que deben compartir la frase
        max_concepts: Máximo de conceptos a retornar

    Returns:
        {frase: [artículos que la contienen}
    """
    import re
    import spacy
    from collections import Counter

    nlp = spacy.load('en_core_web_sm')

    LEAD_WORDS = {
        'a', 'an', 'the', 'its', 'their', 'his', 'her', 'this', 'that',
        'these', 'those', 'some', 'any', 'each', 'every', 'no', 'one', 'two'
    }
    # Palabras genéricas universales: gramática, meta/web y boilerplate de
    # cualquier texto. NO incluye vocabulario de dominio — eso se detecta
    # automáticamente por frecuencia en documentos (ver content_words).
    UNIVERSAL_GENERIC = {
        # Gramática / función
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'while', 'during',
        'before', 'after', 'above', 'below', 'between', 'into', 'through',
        'about', 'against', 'because', 'until', 'unless', 'since', 'although',
        'though', 'whether', 'both', 'either', 'neither', 'each', 'every',
        'few', 'less', 'least', 'much', 'many', 'more', 'most', 'other',
        'others', 'else', 'own', 'same', 'than', 'too', 'very', 'can',
        'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
        'cannot', 'also', 'etc', 'ie', 'eg', 'per', 'via', 'versus', 'vs',
        'including', 'included', 'includes', 'include', 'excluding',
        'excluded', 'according', 'based', 'such', 'like', 'unlike',
        # Hiper-genéricas cross-dominio
        'first', 'second', 'third', 'last', 'new', 'old', 'good', 'bad',
        'big', 'small', 'large', 'long', 'short', 'high', 'low', 'early',
        'late', 'year', 'years', 'time', 'times', 'day', 'days', 'way',
        'ways', 'thing', 'things', 'something', 'anything', 'everything',
        'nothing', 'someone', 'anyone', 'people', 'person', 'man', 'men',
        'woman', 'women', 'child', 'children', 'world', 'life', 'part',
        'parts', 'place', 'places', 'work', 'works', 'number', 'numbers',
        'group', 'groups', 'point', 'points', 'case', 'cases', 'fact',
        'facts', 'example', 'examples', 'question', 'questions', 'answer',
        'answers', 'problem', 'problems', 'idea', 'ideas', 'result',
        'results', 'change', 'changes', 'process', 'processes', 'system',
        'systems', 'information', 'data', 'area', 'areas', 'field',
        'fields', 'level', 'levels', 'type', 'types', 'form', 'forms',
        'line', 'lines', 'end', 'ends', 'side', 'sides', 'name', 'names',
        'word', 'words', 'text', 'texts', 'book', 'books', 'story',
        'stories', 'history', 'various', 'several', 'numerous', 'multiple',
        'certain', 'particular', 'general', 'special', 'main', 'major',
        'minor', 'primary', 'secondary', 'additional', 'extra', 'another',
        'different', 'similar', 'available', 'unavailable', 'current',
        'previous', 'former', 'later', 'earlier', 'subsequent',
        'following', 'preceding', 'original', 'see', 'means', 'mean',
        'order', 'title', 'feature', 'version', 'versions', 'update',
        'updated', 'gameplay', 'storyline', 'universe', 'goal', 'item',
        'items', 'ability', 'abilities', 'character', 'characters',
        # Meta / web / citas
        'article', 'articles', 'wikipedia', 'encyclopedia', 'website',
        'webpage', 'page', 'pages', 'section', 'sections', 'reference',
        'references', 'citation', 'citations', 'external', 'link',
        'links', 'image', 'images', 'figure', 'figures', 'infobox',
        'edit', 'source', 'sources', 'file', 'files', 'category',
        'categories', 'template', 'templates', 'http', 'https', 'html',
        'com', 'org', 'net', 'www', 'archived', 'archive', 'retrieved',
        'retrieve', 'accessed', 'access', 'published', 'publication',
        'author', 'authors', 'volume', 'vol', 'issue', 'issues', 'isbn',
        'doi', 'inc', 'ltd', 'corp', 'corporation', 'department',
        'university', 'press', 'journal', 'journals', 'monthly', 'weekly',
        'daily', 'online', 'copyright', 'rights', 'reserved', 'license',
        'licensed', 'creative', 'commons', 'attribution', 'share',
        'alike', 'language', 'languages', 'translation', 'translated',
        'introduced', 'regional', 'generation', 'january', 'february',
        'whose', 'stronger', 'comprehensive', 'fictional', 'creature',
        'species', 'specific',
        'march', 'april', 'may', 'june', 'july', 'august', 'september',
        'october', 'november', 'december', 'monday', 'tuesday', 'wednesday',
        'thursday', 'friday', 'saturday', 'sunday'
    }
    # Palabras que no deben singularizarse
    NO_SINGULARIZE = {'series', 'species', 'news'}

    def normalize(phrase: str) -> str:
        words = phrase.lower().split()
        while words and words[0] in LEAD_WORDS:
            words = words[1:]
        if not words:
            return ''
        # Unificar variantes verbales frecuentes
        VARIANTS = {'evolved': 'evolution', 'evolve': 'evolution', 'evolves': 'evolution'}
        words = [VARIANTS.get(w, w) for w in words]
        last = words[-1]
        if last not in NO_SINGULARIZE:
            if last.endswith('ies') and len(last) > 4:
                words[-1] = last[:-3] + 'y'  # abilities -> ability
            elif last.endswith('s') and len(last) > 3 \
                    and not last.endswith(('ss', 'us', 'is')):
                words[-1] = last[:-1]
        return ' '.join(words)

    # Quitar acentos para que 'pokémon' -> 'pokemon' coincida en word_df
    import unicodedata

    def strip_accents(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)
        )

    # Frecuencia de palabras individuales (document frequency)
    word_df = Counter()
    for title, text in article_texts.items():
        clean = re.sub(r'\[\d+\]', '', text)
        tokens = set(re.findall(r'[a-zA-Z][a-zA-Z\-]{3,}', clean.lower()))
        for tok in tokens:
            word_df[strip_accents(tok)] += 1

    n_docs = len(article_texts)
    phrase_docs = defaultdict(set)

    for title, text in article_texts.items():
        clean = re.sub(r'\[\d+\]', '', text)
        doc = nlp(clean[:12000])
        seen = set()
        for chunk in doc.noun_chunks:
            phrase = normalize(chunk.text)
            words = phrase.split()
            # Solo frases de 2+ palabras (las de 1 palabra son demasiado genéricas)
            if len(words) < 2 or not (8 <= len(phrase) <= 40) or phrase in seen:
                continue
            # Debe tener al menos una palabra de contenido: no universalmente
            # genérica, de >=4 chars, y que NO aparezca en todos los artículos
            # (word_df == n_docs => boilerplate de este corpus, cualquier dominio)
            content_words = [
                w for w in words
                if w not in UNIVERSAL_GENERIC and len(w) >= 4
                and word_df[strip_accents(w)] < n_docs
            ]
            if not content_words:
                continue
            seen.add(phrase)
            phrase_docs[phrase].add(title)

    # Puntuar: priorizar frases presentes en más artículos (mayor poder
    # de conexión) y desempatar por longitud (frases más específicas)
    article_titles_lower = {t.lower() for t in article_texts}

    def score(phrase: str, dfreq: int) -> float:
        return dfreq * 10 + len(phrase.split())

    candidates = [
        (phrase, docs) for phrase, docs in phrase_docs.items()
        if min_docs <= len(docs) <= n_docs - 1
        and phrase not in article_titles_lower  # excluir títulos de artículos
    ]
    candidates.sort(key=lambda x: -score(x[0], len(x[1])))

    result = {}
    for phrase, docs in candidates[:max_concepts]:
        result[phrase] = sorted(docs)
    return result


def generate_graph_data(chunks_path: str, entities_path: str, transformed_path: str, output_path: str,
                       max_entities_per_article: int = 5, max_concepts_per_article: int = 5,
                       max_nodes: int = 100, max_edges: int = 200):
    """
    Genera los datos del grafo en formato JSON.

    Args:
        chunks_path: Ruta al CSV de chunks
        entities_path: Ruta al JSON de entidades
        transformed_path: Ruta al JSON de artículos transformados
        output_path: Ruta donde guardar el JSON
    """
    # Cargar datos
    chunks_by_article = defaultdict(list)

    # Cargar chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            article = row['article_title']
            chunk_text = row['chunk_text']  # Texto completo del chunk
            chunks_by_article[article].append({
                'text': chunk_text,
                'chunk_id': row.get('chunk_id', 0)
            })

    # Cargar entidades y conceptos
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
        entities = entities_data.get('entities', {})
        concepts = entities_data.get('concepts', {})

    # Filtrar y limitar entidades y conceptos
    print("Filtrando entidades y conceptos...")
    filtered_entities = {}
    filtered_concepts = {}

    # Contar frecuencia de entidades y conceptos
    entity_freq = defaultdict(int)
    concept_freq = defaultdict(int)

    for article, article_entities in entities.items():
        for entity_type, entity_list in article_entities.items():
            for entity in entity_list:
                entity_freq[entity] += 1

    for article, article_concepts in concepts.items():
        for concept in article_concepts:
            concept_freq[concept] += 1

    # Filtrar entidades: solo las que aparecen en al menos 2 artículos o son muy frecuentes
    min_entity_freq = 2
    for article, article_entities in entities.items():
        filtered_article_entities = {}
        for entity_type, entity_list in article_entities.items():
            # Filtrar por frecuencia y longitud
            filtered_list = [e for e in entity_list
                           if entity_freq[e] >= min_entity_freq and len(e) > 3]
            # Limitar por artículo
            filtered_list = filtered_list[:max_entities_per_article]
            if filtered_list:
                filtered_article_entities[entity_type] = filtered_list
        if filtered_article_entities:
            filtered_entities[article] = filtered_article_entities

    # Filtrar conceptos: solo los más frecuentes y relevantes
    for article, article_concepts in concepts.items():
        # Filtrar por frecuencia y longitud
        filtered_list = [c for c in article_concepts
                        if concept_freq[c] >= 2 and len(c) > 4]
        # Limitar por artículo
        filtered_list = filtered_list[:max_concepts_per_article]
        if filtered_list:
            filtered_concepts[article] = filtered_list

    entities = filtered_entities
    concepts = filtered_concepts
    print(f"  Entidades filtradas: {sum(len(v) for v in entities.values())}")
    print(f"  Conceptos filtrados: {sum(len(v) for v in concepts.values())}")

    # Cargar títulos de artículos (ordenar para que el principal esté primero)
    with open(transformed_path, 'r', encoding='utf-8') as f:
        transformed_data = json.load(f)
        article_titles = []
        if isinstance(transformed_data, list):
            for article in transformed_data:
                if isinstance(article, dict) and 'title' in article:
                    article_titles.append(article['title'])
        elif isinstance(transformed_data, dict):
            # Si es diccionario, las claves son títulos
            article_titles = list(transformed_data.keys())

        # Si hay un artículo principal conocido, ponerlo primero
        # Por ahora usamos el orden original, pero se puede mejorar

    # Calcular similitud entre artículos ANTES de crear nodos,
    # para filtrar los artículos poco relevantes
    print("Calculando similitud entre artículos...")

    # Crear documentos por artículo (combinar chunks)
    article_texts = {}
    for title in article_titles:
        chunks = chunks_by_article.get(title, [])
        if chunks:
            article_texts[title] = ' '.join([chunk['text'] for chunk in chunks[:4]])[:8000]

    # Filtrar artículos: conservar solo los más relevantes respecto al principal
    MAX_RELATED_ARTICLES = 7   # Artículos relacionados máximos en el grafo
    kept_titles = list(article_titles)
    relevance_scores = {}

    if len(article_texts) > 1:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        sim_titles = [t for t in article_titles if t in article_texts]
        texts = [article_texts[t] for t in sim_titles]

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            main_article = sim_titles[0] if sim_titles else article_titles[0]
            main_idx = sim_titles.index(main_article) if main_article in sim_titles else 0

            # Puntaje = similitud TF-IDF al principal + bonus por posición
            # (el pipeline ya ordena por relevancia según Wikipedia 'morelike:')
            n = len(sim_titles)
            scored = []
            for i, t in enumerate(sim_titles):
                if t == main_article:
                    continue
                sim = float(similarity_matrix[main_idx][i])
                position_bonus = 1.5 - (i / max(n - 1, 1))  # 1.5 temprano -> 0.5 tardío
                score = sim * position_bonus
                scored.append((t, sim, score))
                relevance_scores[t] = score

            scored.sort(key=lambda x: -x[2])
            kept_related = [t for t, _, _ in scored[:MAX_RELATED_ARTICLES]]

            # Mantener el orden original del pipeline para títulos conservados
            kept_set = set(kept_related) | {main_article}
            kept_titles = [t for t in article_titles if t in kept_set]

            print(f"  Artículo principal: {main_article}")
            print("  Relevancia de artículos relacionados:")
            for t, sim, score in scored:
                mark = "OK" if t in kept_set else "--"
                print(f"    {mark} {t}: sim={sim:.3f} score={score:.3f}")
            dropped = [t for t in article_titles if t not in kept_set]
            if dropped:
                print(f"  Artículos descartados por baja relevancia: {dropped}")
        except Exception as e:
            print(f"  Error calculando similitud: {e}")

    # Construir nodos y aristas
    nodes = []
    edges = []
    node_id_map = {}

    # Nodos para artículos (el primero es el principal)
    for i, title in enumerate(kept_titles):
        node_id = f'article_{i}'
        node_id_map[title] = node_id

        # Colores: principal dorado, relacionados gris/azul
        if title == kept_titles[0]:
            color = '#FFD700'  # Dorado para artículo principal
        else:
            color = '#61AFEF'  # Azul para artículos relacionados

        # Tamaño proporcional a la relevancia (principal = 16)
        if title == kept_titles[0]:
            size = 16
        else:
            score = relevance_scores.get(title, 0)
            max_score = max(relevance_scores.values()) if relevance_scores else 1
            size = 8 + int(4 * (score / max_score)) if max_score > 0 else 10

        nodes.append({
            'id': node_id,
            'label': title,
            'type': 'article',
            'color': color,
            'size': size,
            'chunks': [chunk['text'] for chunk in chunks_by_article.get(title, [])[:5]]
        })

    # Nodos de concepto: frases compartidas entre artículos (ej: "Mega Evolution")
    # Usar más chunks por artículo para detectar conceptos que aparecen más adelante
    kept_texts = {
        t: ' '.join(c['text'] for c in chunks_by_article.get(t, [])[:8])
        for t in kept_titles if chunks_by_article.get(t)
    }
    shared_concepts = {}
    if len(kept_texts) >= 3:
        try:
            shared_concepts = extract_shared_concepts(kept_texts, min_docs=2, max_concepts=6)
            print(f"  Conceptos compartidos detectados: {list(shared_concepts.keys())}")
        except Exception as e:
            print(f"  Error extrayendo conceptos: {e}")

    concept_nodes = []
    concept_members = []
    for ci, (phrase, member_articles) in enumerate(shared_concepts.items()):
        concept_id = f'concept_{ci}'
        label = phrase.title()
        concept_nodes.append({
            'id': concept_id,
            'label': label,
            'type': 'concept',
            'color': '#98C379',  # Verde para conceptos
            'size': 7,
            'chunks': [],
            'members': member_articles
        })
        concept_members.append(member_articles)
        nodes.append(concept_nodes[-1])

    # Nodos para entidades (DESHABILITADO - no necesarios para el grafo)
    entity_id_map = {}
    # entity_counter = 0
    # for article, article_entities in entities.items():
    #     for entity_type, entity_list in article_entities.items():
    #         for entity in entity_list:
    #             if entity not in entity_id_map and len(entity) > 3 and entity_counter < max_nodes // 3:
    #                 node_id = f'entity_{entity_counter}'
    #                 entity_id_map[entity] = node_id
    #                 entity_counter += 1
    #                 nodes.append({
    #                     'id': node_id,
    #                     'label': entity,
    #                     'type': 'entity',
    #                     'color': '#98C379',  # Verde del tema One Dark
    #                     'size': 6,
    #                     'entity_type': entity_type,
    #                     'chunks': []
    #                 })

    # Nodos para conceptos (DESHABILITADO - no necesarios para el grafo)
    concept_id_map = {}
    # concept_counter = 0
    # for article, article_concepts in concepts.items():
    #     for concept in article_concepts:
    #         if concept not in concept_id_map and len(concept) > 4 and concept_counter < max_nodes // 3:
    #             node_id = f'concept_{concept_counter}'
    #             concept_id_map[concept] = node_id
    #             concept_counter += 1
    #             nodes.append({
    #                 'id': node_id,
    #                 'label': concept,
    #                 'type': 'concept',
    #                 'color': '#61AFEF',  # Azul del tema One Dark
    #                 'size': 6,
    #                 'chunks': []
    #             })

    # Aristas artículo-entidad (DESHABILITADO - no necesarios para el grafo)
    edge_set = set()
    edge_count = 0
    # for article, article_entities in entities.items():
    #     if article in node_id_map and edge_count < max_edges // 2:
    #         for entity_type, entity_list in article_entities.items():
    #             for entity in entity_list:
    #                 if entity in entity_id_map and edge_count < max_edges // 2:
    #                     edge_key = (node_id_map[article], entity_id_map[entity])
    #                     if edge_key not in edge_set:
    #                         edges.append({
    #                             'from': node_id_map[article],
    #                             'to': entity_id_map[entity],
    #                             'label': entity_type,
    #                             'type': 'article_entity'
    #                         })
    #                         edge_set.add(edge_key)
    #                         edge_count += 1

    # Aristas artículo-concepto (DESHABILITADO - no necesarios para el grafo)
    # for article, article_concepts in concepts.items():
    #     if article in node_id_map and edge_count < max_edges:
    #         for concept in article_concepts:
    #             if concept in concept_id_map and edge_count < max_edges:
    #                 edge_key = (node_id_map[article], concept_id_map[concept])
    #                 if edge_key not in edge_set:
    #                     edges.append({
    #                         'from': node_id_map[article],
    #                         'to': concept_id_map[concept],
    #                         'label': 'concept',
    #                         'type': 'article_concept'
    #                     })
    #                     edge_set.add(edge_key)
    #                     edge_count += 1

    # Aristas entre artículos basadas en similitud de contenido (TF-IDF)
    edge_set = set()
    edge_count = 0

    if len(kept_titles) > 1:
        # Crear matriz TF-IDF solo para artículos conservados
        vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
        edge_titles = [t for t in kept_titles if t in article_texts]
        texts = [article_texts[t] for t in edge_titles]

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # PASO 1: Conectar el artículo principal con todos los demás conservados
            main_article = kept_titles[0]
            for art in kept_titles[1:]:
                if main_article in node_id_map and art in node_id_map:
                    edge_key = (node_id_map[main_article], node_id_map[art])
                    if edge_key not in edge_set and edge_count < max_edges:
                        edges.append({
                            'from': node_id_map[main_article],
                            'to': node_id_map[art],
                            'type': 'main_connection',
                            'width': 2
                        })
                        edge_set.add(edge_key)
                        edge_count += 1

            # PASO 2: Aristas de similitud entre artículos (umbral 0.25)
            similarity_threshold = 0.25
            similarity_edges = 0
            for i, art1 in enumerate(edge_titles):
                for j, art2 in enumerate(edge_titles):
                    if i < j:
                        similarity = similarity_matrix[i][j]
                        if similarity > similarity_threshold and edge_count < max_edges:
                            if art1 in node_id_map and art2 in node_id_map:
                                edge_key = (node_id_map[art1], node_id_map[art2])
                                if edge_key not in edge_set:
                                    edges.append({
                                        'from': node_id_map[art1],
                                        'to': node_id_map[art2],
                                        'type': 'article_similarity',
                                        'width': min(similarity * 3, 2)
                                    })
                                    edge_set.add(edge_key)
                                    edge_count += 1
                                    similarity_edges += 1

            print(f"  Conexiones principales: {len(kept_titles) - 1}")
            print(f"  Aristas de similitud (>{similarity_threshold}): {similarity_edges}")
            print(f"  Total de aristas: {edge_count}")
        except Exception as e:
            print(f"  Error calculando similitud: {e}")

    # Aristas artículo-concepto (frases compartidas)
    concept_edge_count = 0
    for concept_node, member_articles in zip(concept_nodes, concept_members):
        for art in member_articles:
            if art in node_id_map and edge_count < max_edges:
                edge_key = (node_id_map[art], concept_node['id'])
                if edge_key not in edge_set:
                    edges.append({
                        'from': node_id_map[art],
                        'to': concept_node['id'],
                        'type': 'article_concept',
                        'width': 1
                    })
                    edge_set.add(edge_key)
                    edge_count += 1
                    concept_edge_count += 1

    if concept_edge_count:
        print(f"  Aristas artículo-concepto: {concept_edge_count}")
        print(f"  Total de aristas: {edge_count}")

    # Guardar datos
    graph_data = {
        'nodes': nodes,
        'edges': edges
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"OK Datos del grafo generados: {output_path}")
    print(f"  - Nodos: {len(nodes)}")
    print(f"  - Aristas: {len(edges)}")


def main():
    """Función principal."""
    base_dir = Path(__file__).parent
    chunks_path = base_dir / "data" / "processed" / "chunks_dataframe.csv"
    entities_path = base_dir / "data" / "processed" / "entities.json"
    transformed_path = base_dir / "data" / "processed" / "transformed_articles.json"
    output_path = base_dir / "data" / "processed" / "graph_data.json"

    # Verificar que existan los archivos
    if not all(p.exists() for p in [chunks_path, entities_path, transformed_path]):
        print("ERROR: No se encontraron los archivos procesados.")
        print("Primero ejecuta el pipeline de conocimiento.")
        return

    # Generar con límites mucho más restrictivos para mejor visualización
    generate_graph_data(
        str(chunks_path),
        str(entities_path),
        str(transformed_path),
        str(output_path),
        max_entities_per_article=2,  # Solo 2 entidades por artículo
        max_concepts_per_article=2,  # Solo 2 conceptos por artículo
        max_nodes=40,                # Solo 40 nodos totales
        max_edges=60                 # Solo 60 aristas totales
    )


if __name__ == "__main__":
    main()
