"""
Servidor web simple para la aplicación de explorador de conocimiento.
Sirve la aplicación HTML y la API para preguntas.
"""

import json
import csv
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import subprocess

# Configurar encoding UTF-8 para Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Importar el generador de Ollama
sys.path.append(str(Path(__file__).parent / "src" / "generation"))
from ollama_generator import OllamaGenerator


# Singleton para el generador
_generator = None
_generator_mtime = None


def get_generator():
    """Obtiene o crea el generador singleton.

    Recarga automáticamente si los datos procesados cambiaron
    (por ejemplo, después de procesar un nuevo tema).
    """
    global _generator, _generator_mtime

    base_dir = Path(__file__).parent
    chunks_path = base_dir / "data" / "processed" / "chunks_dataframe.csv"
    embeddings_path = base_dir / "data" / "processed" / "chunks_embeddings.json"
    entities_path = base_dir / "data" / "processed" / "entities.json"

    # Detectar si los datos cambiaron desde la última carga
    current_mtime = chunks_path.stat().st_mtime if chunks_path.exists() else None
    if _generator is not None and current_mtime == _generator_mtime:
        return _generator

    if not all(p.exists() for p in [chunks_path, embeddings_path, entities_path]):
        raise FileNotFoundError("No se encontraron los datos procesados. Procesa un tema primero.")

    _generator = OllamaGenerator(model_name="qwen3:4b-instruct-2507-q4_K_M")
    _generator.load_data(str(chunks_path), str(embeddings_path), str(entities_path))
    _generator_mtime = current_mtime
    return _generator


class KnowledgeHandler(SimpleHTTPRequestHandler):
    """Handler personalizado para servir la aplicación y la API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        """Maneja peticiones GET."""
        parsed_path = urlparse(self.path)

        # API endpoint para preguntas
        if parsed_path.path == '/api/ask':
            self.handle_ask_request(parse_qs(parsed_path.query))
            return

        if parsed_path.path == '/api/process_topic':
            params = parse_qs(parsed_path.query)
            topic = params.get('topic', [''])[0]
            self.handle_process_topic(topic)
            return

        # Servir archivos estáticos
        super().do_GET()

    def do_POST(self):
        """Maneja peticiones POST."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/api/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            self.handle_ask_post(data)
            return

        if parsed_path.path == '/api/upload':
            self.handle_file_upload()
            return

        self.send_error(404)

    def handle_ask_request(self, params):
        """Maneja peticiones GET para preguntas."""
        query = params.get('q', [''])[0]
        if not query:
            self.send_error(400, "Missing query parameter 'q'")
            return

        result = self.process_question(query)
        self.send_json_response(result)

    def handle_ask_post(self, data):
        """Maneja peticiones POST para preguntas."""
        query = data.get('query', '')
        if not query:
            self.send_json_response({'error': 'Missing query'})
            return

        result = self.process_question(query)
        self.send_json_response(result)

    def handle_process_topic(self, topic):
        """Procesa un tema usando el pipeline existente."""
        try:
            if not topic:
                self.send_json_response({'error': 'No topic provided'})
                return

            print(f"Procesando tema: {topic}")

            # Ejecutar el pipeline con el tema
            base_dir = Path(__file__).parent
            pipeline_path = base_dir / "pipeline.py"

            # Ejecutar pipeline
            result = subprocess.run(
                [sys.executable, str(pipeline_path)],
                input=f"{topic}\n",  # Tema + enter para defaults
                stdout=subprocess.DEVNULL,  # No capturar stdout
                stderr=subprocess.PIPE,    # Capturar errores
                text=True,
                cwd=str(base_dir),
                timeout=900  # 15 minutos timeout
            )

            if result.returncode != 0:
                print(f"Error en pipeline: {result.stderr}")
                self.send_json_response({'error': f'Pipeline failed: {result.stderr}'})
                return

            # Regenerar datos del grafo
            print("Regenerando datos del grafo...")
            generate_script = base_dir / "generate_graph_data.py"
            graph_result = subprocess.run(
                [sys.executable, str(generate_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(base_dir)
            )

            if graph_result.returncode != 0:
                print(f"Error generando grafo: {graph_result.stderr}")
                self.send_json_response({'error': f'Graph generation failed: {graph_result.stderr}'})
                return

            self.send_json_response({
                'success': True,
                'message': f'Tema "{topic}" procesado exitosamente',
                'topic': topic
            })

        except subprocess.TimeoutExpired:
            self.send_json_response({'error': 'Processing timed out'})
        except Exception as e:
            print(f"Error procesando tema: {e}")
            self.send_json_response({'error': str(e)})

    def handle_file_upload(self):
        """Maneja la subida de archivos."""
        try:
            # Parsear multipart/form-data
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self.send_json_response({'error': 'Invalid content type'})
                return

            # Leer el body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # Parsear archivos (simplificado)
            files_processed = []

            # Guardar archivos en directorio temporal
            import tempfile
            import os

            temp_dir = tempfile.mkdtemp()
            boundary = content_type.split('boundary=')[1].encode()

            # Parsear partes del multipart
            parts = body.split(b'--' + boundary)

            for part in parts:
                if b'filename=' in part:
                    # Extraer nombre del archivo (sanitizar para evitar path traversal)
                    import os
                    filename_start = part.find(b'filename="') + 10
                    filename_end = part.find(b'"', filename_start)
                    filename = part[filename_start:filename_end].decode('utf-8')
                    filename = os.path.basename(filename)  # Solo el nombre, sin rutas

                    # Extraer contenido
                    content_start = part.find(b'\r\n\r\n') + 4
                    content_end = part.rfind(b'\r\n--')
                    content = part[content_start:content_end]

                    # Guardar archivo
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(content)

                    # Procesar archivo (simulado por ahora)
                    files_processed.append({
                        'name': filename,
                        'chunks': 10,  # Simulado
                        'path': file_path
                    })

            self.send_json_response({
                'success': True,
                'files': files_processed,
                'message': f'{len(files_processed)} archivos procesados'
            })

        except Exception as e:
            print(f"Error procesando archivos: {e}")
            self.send_json_response({'error': str(e)})

    def process_question(self, query):
        """Procesa una pregunta usando Ollama."""
        try:
            generator = get_generator()

            # Procesar pregunta
            result = generator.answer_question(query)

            # Extraer artículos relevantes de los chunks
            relevant_articles = list(set([chunk.get('article', '') for chunk in result.get('chunks', [])]))

            return {
                'answer': result['answer'],
                'concepts': result.get('concepts', []),
                'entities': result.get('entities', {}),
                'chunks': result.get('chunks', []),
                'relevant_articles': relevant_articles,
                'source': result.get('source', 'corpus')
            }

        except Exception as e:
            print(f"Error procesando pregunta: {e}")
            return {
                'error': str(e),
                'answer': 'Ocurrió un error al procesar tu pregunta.'
            }

    def send_json_response(self, data):
        """Envía una respuesta JSON."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(json.dumps(data, ensure_ascii=False).encode('utf-8'))))
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        """Personaliza el log de mensajes."""
        print(f"{self.address_string()} - {format % args}")


def warmup_generator():
    """Carga el generador en segundo plano para que la primera pregunta sea rápida."""
    try:
        get_generator()
        print("Generador listo.")
    except Exception as e:
        print(f"Error en warmup del generador: {e}")


def main():
    """Función principal del servidor."""
    port = 8000
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, KnowledgeHandler)

    print(f"=" * 60)
    print(f"Servidor iniciado en http://localhost:{port}")
    print(f"=" * 60)
    print(f"\nAplicación disponible en:")
    print(f"  http://localhost:{port}/app.html")
    print(f"\nAPI disponible en:")
    print(f"  http://localhost:{port}/api/ask?q=tu_pregunta")
    print(f"\nPresiona Ctrl+C para detener el servidor")
    print(f"=" * 60)

    # Cargar generador en segundo plano
    import threading
    threading.Thread(target=warmup_generator, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServidor detenido.")
        httpd.server_close()


if __name__ == "__main__":
    main()
