from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .dashboard import render_dashboard
from .state import PhaseOneRuntime


def create_server(runtime: PhaseOneRuntime, host: str, port: int) -> ThreadingHTTPServer:
    handler = _make_handler(runtime)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    host = os.getenv("MII_HOST", "127.0.0.1")
    port = int(os.getenv("MII_PORT", "8080"))

    runtime = PhaseOneRuntime()
    runtime.run_background(interval_seconds=1.0)
    server = create_server(runtime, host, port)
    print(f"ML Incident Intelligence Phase 1 listening on http://{host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop_background()
        server.server_close()


def _make_handler(runtime: PhaseOneRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                self._send_html(render_dashboard(runtime.snapshot()))
                return
            if self.path == "/api/state":
                self._send_json(runtime.snapshot().to_dict())
                return
            if self.path == "/api/graph":
                self._send_json(runtime.snapshot().dependency_graph.to_dict())
                return
            if self.path == "/api/rca":
                self._send_json(
                    {
                        "deployment_events": [event.to_dict() for event in runtime.snapshot().deployment_events],
                        "root_cause_hypotheses": [hypothesis.to_dict() for hypothesis in runtime.snapshot().root_cause_hypotheses],
                    }
                )
                return
            if self.path == "/healthz":
                self._send_json({"status": "ok", "step": runtime.step})
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


if __name__ == "__main__":
    main()
