"""
Lightweight background HTTP server for OBS Studio Browser Source.
Serves overlay/ directory on http://127.0.0.1:8080 without CORS issues.
"""

import http.server
import socketserver
import threading
from pathlib import Path
from typing import Optional

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Suppresses standard GET log output to keep console tidy."""
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        # Allow cross-origin and disable browser cache for real-time OBS updates
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

class OverlayServer:
    def __init__(self, directory: Path, port: int = 8080):
        self.directory = directory
        self.port = port
        self.httpd: Optional[socketserver.TCPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the HTTP server on a background daemon thread."""
        if self.server_thread and self.server_thread.is_alive():
            return

        def handler(*args, **kwargs):
            return QuietHTTPRequestHandler(*args, directory=str(self.directory), **kwargs)

        try:
            socketserver.TCPServer.allow_reuse_address = True
            self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), handler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"[OverlayServer] OBS Stream Overlay live at: http://localhost:{self.port}/")
        except Exception as e:
            print(f"[OverlayServer] Notice: Could not bind port {self.port} ({e}).")

    def stop(self):
        """Stops the HTTP server."""
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        print("[OverlayServer] Stopped.")
