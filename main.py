"""
Script principal para ejecutar el pipeline completo del proyecto NLP.
Orquesta la extracción de datos, EDA, transformación y análisis semántico.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Configurar encoding UTF-8 para Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Agregar directorios al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_extraction'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'transformation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'generation'))


def run_script(script_path: str, description: str):
    """
    Ejecuta un script de Python.

    Args:
        script_path: Ruta al script a ejecutar
        description: Descripción del script para mostrar
    """
    print(f"\n{'='*60}")
    print(f"Ejecutando: {description}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error ejecutando {description}: {e}")
        return False


def check_dependencies():
    """Verifica que las dependencias principales estén instaladas."""
    # Nota: scikit-learn se importa como 'sklearn' en Python
    required_packages = ['requests', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                        'wordcloud', 'nltk', 'spacy', 'sklearn', 'bs4']

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("ERROR: Las siguientes dependencias no están instaladas:")
        for package in missing:
            # Mapear nombre de módulo a nombre de paquete en requirements.txt
            package_names = {
                'sklearn': 'scikit-learn'
            }
            display_name = package_names.get(package, package)
            print(f"  - {display_name}")
        print("\nEjecuta: pip install -r requirements.txt")
        return False

    return True


def check_spacy_model():
    """Verifica que el modelo de spaCy esté instalado."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        print("WARNING: Modelo de spaCy no encontrado.")
        print("Ejecuta: python -m spacy download en_core_web_sm")
        return False


def show_menu():
    """Muestra el menú principal."""
    print("\n" + "="*60)
    print("PIPELINE DE CONOCIMIENTO")
    print("="*60)
    print("\nOpciones:")
    print("1. Ejecutar pipeline de conocimiento")
    print("2. Modo interactivo (preguntas y respuestas con Ollama)")
    print("3. Salir")
    print()


def main():
    """Función principal."""
    # Verificar dependencias
    if not check_dependencies():
        return

    # Verificar modelo de spaCy (opcional pero recomendado)
    check_spacy_model()

    base_dir = Path(__file__).parent

    while True:
        show_menu()
        choice = input("Selecciona una opción (1-3): ").strip()

        if choice == "1":
            # Pipeline de conocimiento
            run_script(str(base_dir / 'pipeline.py'), "Pipeline de conocimiento")

        elif choice == "2":
            # Modo interactivo con Ollama
            run_script(str(base_dir / 'src' / 'generation' / 'ollama_generator.py'), "Modo interactivo con Ollama")

        elif choice == "3":
            print("\nSaliendo...")
            break

        else:
            print("\nOpción no válida. Intenta de nuevo.")

        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()
