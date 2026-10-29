"""HTTP-сервис без внешних зависимостей.

POST /solve        тело — картинка (png/jpg/gif как есть)
                   или JSON {"image": "<base64 или data:image/...;base64,...>"}
GET  /health       {"ok": true, "version": ...}
Ответ /solve: {"text": "...", "confidence": 0.998, "ms": 2.9}
"""
from __future__ import annotations

import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .solver import Solver

MAX_BODY = 2 * 1024 * 1024


def _image_from_request(body: bytes, ctype: str) -> bytes:
    if "json" in ctype or body[:1] == b"{":
        data = json.loads(body.decode("utf-8"))["image"]
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
    return body


def make_handler(solver: Solver):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"kapcherka/{__version__}"

        def _send(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "version": __version__})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/solve":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                self._send(400, {"error": "пустое или слишком большое тело запроса"})
                return
            body = self.rfile.read(n)
            try:
                img = _image_from_request(body, self.headers.get("Content-Type", ""))
                t = time.perf_counter()
                r = solver.solve(img)
                ms = (time.perf_counter() - t) * 1000
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": f"не удалось прочитать картинку: {e}"})
                return
            self._send(200, {"text": r.text, "confidence": round(r.confidence, 4), "ms": round(ms, 2)})

        def log_message(self, fmt, *args):  # тихий режим
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, model=None, threads: int = 1) -> None:
    solver = Solver(model, threads=threads)
    httpd = ThreadingHTTPServer((host, port), make_handler(solver))
    print(f"kapcherka {__version__}: http://{host}:{port}  (POST /solve, GET /health)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
