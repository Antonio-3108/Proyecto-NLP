import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re


class WikipediaEDA:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.articles = self._load_data()
        self.df = self._create_dataframe()
    
    def _load_data(self) -> list:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.articles)
    
    def basic_statistics(self):
        print("=" * 60)
        print("ESTADÍSTICAS BÁSICAS")
        print("=" * 60)
        
        print(f"\nTotal de artículos: {len(self.df)}")
        print(f"\nColumnas disponibles: {list(self.df.columns)}")
        
        # Estadísticas de longitud
        self.df['text_length'] = self.df['extract'].str.len()
        self.df['word_count'] = self.df['extract'].str.split().str.len()
        
        print(f"\n--- Longitud del texto (caracteres) ---")
        print(self.df['text_length'].describe())
        
        print(f"\n--- Conteo de palabras ---")
        print(self.df['word_count'].describe())
        
        # Categorías
        total_categories = sum(len(cats) for cats in self.df['categories'])
        print(f"\n--- Categorías ---")
        print(f"Total de categorías: {total_categories}")
        print(f"Promedio por artículo: {total_categories/len(self.df):.1f}")
        
        # Links
        total_links = sum(len(links) for links in self.df['links'])
        print(f"\n--- Links internos ---")
        print(f"Total de links: {total_links}")
        print(f"Promedio por artículo: {total_links/len(self.df):.1f}")
    
    def visualize_length_distribution(self, output_dir: str = "data/processed"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribución de caracteres
        axes[0].hist(self.df['text_length'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Longitud (caracteres)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Longitud de Artículos (caracteres)')
        axes[0].grid(True, alpha=0.3)
        
        # Distribución de palabras
        axes[1].hist(self.df['word_count'], bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Conteo de palabras')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Longitud de Artículos (palabras)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"{output_dir}/length_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {output_path}")
        plt.close()
    
    def analyze_vocabulary(self, output_dir: str = "data/processed"):
        """Analiza el vocabulario del corpus."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Combinar todo el texto
        all_text = ' '.join(self.df['extract'].tolist())
        
        # Tokenización simple
        words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        word_freq = Counter(words)
        
        print("\n" + "=" * 60)
        print("ANÁLISIS DE VOCABULARIO")
        print("=" * 60)
        
        print(f"\nTotal de palabras: {len(words)}")
        print(f"Vocabulario único: {len(word_freq)}")
        print(f"Palabras únicas / Total: {len(word_freq)/len(words)*100:.2f}%")
        
        print(f"\n--- 20 palabras más frecuentes ---")
        for word, freq in word_freq.most_common(20):
            print(f"{word:20s} : {freq:5d}")
        
        # Visualizar palabras más frecuentes
        plt.figure(figsize=(12, 6))
        top_words = word_freq.most_common(30)
        words_list = [w[0] for w in top_words]
        counts = [w[1] for w in top_words]
        
        plt.barh(words_list[::-1], counts[::-1])
        plt.xlabel('Frecuencia')
        plt.title('30 Palabras Más Frecuentes')
        plt.tight_layout()
        
        output_path = f"{output_dir}/word_frequency.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {output_path}")
        plt.close()
    
    def analyze_categories(self, output_dir: str = "data/processed"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Contar categorías
        all_categories = []
        for cats in self.df['categories']:
            all_categories.extend(cats)
        
        category_freq = Counter(all_categories)
        
        print("\n" + "=" * 60)
        print("ANÁLISIS DE CATEGORÍAS")
        print("=" * 60)
        
        print(f"\nTotal de categorías únicas: {len(category_freq)}")
        
        print(f"\n--- 15 categorías más comunes ---")
        for cat, freq in category_freq.most_common(15):
            print(f"{cat:50s} : {freq:2d}")
    
    def generate_report(self, output_dir: str = "data/processed"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE COMPLETO")
        print("=" * 60)
        
        self.basic_statistics()
        self.visualize_length_distribution(output_dir)
        self.analyze_vocabulary(output_dir)
        self.analyze_categories(output_dir)
        
        # Guardar DataFrame procesado
        df_path = f"{output_dir}/articles_dataframe.csv"
        self.df.to_csv(df_path, index=False, encoding='utf-8')
        print(f"\nDataFrame guardado en: {df_path}")
        
        print("\n" + "=" * 60)
        print("EDA COMPLETADO")
        print("=" * 60)


def main():
    data_path = "data/raw/wikipedia_articles.json"
    
    if not Path(data_path).exists():
        print(f"No se encontró el archivo: {data_path}")
        print("Primero ejecutar el script de extracción de datos.")
        return
    
    eda = WikipediaEDA(data_path)
    eda.generate_report()


if __name__ == "__main__":
    main()