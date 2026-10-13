"""Local dev server: python tools/dev_server.py  ->  http://localhost:8000

Serves the repo root and generates /nmrdash-files.json on the fly (CI does this at deploy).
"""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/nmrdash-files.json":
            body = json.dumps(sorted(p.name for p in (ROOT / "nmrdash").glob("*.py"))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print("http://localhost:8000")
    ThreadingHTTPServer(("localhost", 8000), H).serve_forever()
