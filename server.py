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


def _clear_data_dirs():
    """Elimina todo el contenido de data/uploads, data/raw y data/processed."""
    base_dir = Path(__file__).parent
    deleted = 0
    for data_subdir in (base_dir / "data" / "uploads",
                        base_dir / "data" / "raw",
                        base_dir / "data" / "processed"):
        if data_subdir.exists():
            for f in data_subdir.iterdir():
                if f.is_file():
                    f.unlink()
                    deleted += 1
    return deleted


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

        if parsed_path.path == '/api/process_uploads':
            self.handle_process_uploads()
            return

        if parsed_path.path == '/api/upload':
            self.handle_file_upload()
            return

        if parsed_path.path == '/api/reset':
            self.handle_reset()
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
                # stdout heredado: el progreso del pipeline se ve en la consola
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
        """Guarda los archivos subidos en data/uploads/ para procesarlos."""
        try:
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self.send_json_response({'error': 'Invalid content type'})
                return

            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            import os
            uploads_dir = Path(__file__).parent / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            boundary = content_type.split('boundary=')[1].encode()
            parts = body.split(b'--' + boundary)

            saved_files = []
            supported = {'.txt', '.md', '.pdf'}

            for part in parts:
                if b'filename=' not in part:
                    continue

                # Extraer nombre del archivo (sanitizar para evitar path traversal)
                filename_start = part.find(b'filename="') + 10
                filename_end = part.find(b'"', filename_start)
                filename = part[filename_start:filename_end].decode('utf-8')
                filename = os.path.basename(filename)

                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported:
                    print(f"  Archivo no soportado, se omite: {filename}")
                    continue

                # Extraer contenido
                content_start = part.find(b'\r\n\r\n') + 4
                content_end = part.rfind(b'\r\n--')
                content = part[content_start:content_end]

                # Guardar en data/uploads (evitar sobrescribir)
                file_path = uploads_dir / filename
                counter = 1
                while file_path.exists():
                    stem, e = os.path.splitext(filename)
                    file_path = uploads_dir / f"{stem}_{counter}{e}"
                    counter += 1

                with open(file_path, 'wb') as f:
                    f.write(content)

                saved_files.append(file_path.name)
                print(f"  Archivo guardado: {file_path.name} ({len(content)} bytes)")

            if not saved_files:
                self.send_json_response({'error': 'No se encontraron archivos válidos (.txt, .md, .pdf)'})
                return

            self.send_json_response({
                'success': True,
                'files': saved_files,
                'message': f'{len(saved_files)} archivo(s) guardado(s) en data/uploads/'
            })

        except Exception as e:
            print(f"Error procesando archivos: {e}")
            self.send_json_response({'error': str(e)})

    def handle_reset(self):
        """Elimina uploads, datos raw y procesados (nueva exploración)."""
        try:
            deleted = _clear_data_dirs()
            print(f"Reset completo: {deleted} archivos eliminados")
            self.send_json_response({
                'success': True,
                'message': 'Datos eliminados. Listo para una nueva exploración.'
            })

        except Exception as e:
            print(f"Error en reset: {e}")
            self.send_json_response({'error': str(e)})

    def handle_process_uploads(self):
        """Ejecuta process_uploads.py para fusionar documentos con el corpus."""
        try:
            base_dir = Path(__file__).parent
            script_path = base_dir / "process_uploads.py"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(base_dir),
                timeout=600
            )

            if result.returncode != 0:
                print(f"Error procesando uploads: {result.stderr}")
                self.send_json_response({'error': f'Upload processing failed: {result.stderr}'})
                return

            # Regenerar datos del grafo
            print("Regenerando datos del grafo...")
            generate_script = base_dir / "generate_graph_data.py"
            graph_result = subprocess.run(
                [sys.executable, str(generate_script)],
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
                'message': 'Documentos procesados exitosamente'
            })

        except subprocess.TimeoutExpired:
            self.send_json_response({'error': 'Upload processing timed out'})
        except Exception as e:
            print(f"Error procesando uploads: {e}")
            self.send_json_response({'error': str(e)})

    def process_question(self, query):
        """Procesa una pregunta usando Ollama."""
        import time
        start = time.time()
        try:
            generator = get_generator()

            # Procesar pregunta
            result = generator.answer_question(query)
            elapsed = time.time() - start
            print(f"[/api/ask] '{query[:60]}' -> {elapsed:.1f}s | source={result.get('source')} | top_sim={result.get('top_similarity', 0):.3f}")

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
        """Log de requests HTTP con timestamp."""
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {self.address_string()} - {format % args}")


def warmup_generator():
    """Carga el generador en segundo plano para que la primera pregunta sea rápida."""
    try:
        get_generator()
        print("Generador listo.")
    except Exception as e:
        print(f"Error en warmup del generador: {e}")


def main():
    """Función principal del servidor."""
    # Limpiar datos de sesiones anteriores: cada inicio del servidor
    # comienza con una base de conocimiento vacía
    deleted = _clear_data_dirs()
    if deleted:
        print(f"Inicio: {deleted} archivos de datos anteriores eliminados")

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
