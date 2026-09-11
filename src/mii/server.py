from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

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


def _investigate_incident(runtime: PhaseOneRuntime, incident) -> dict[str, object]:
    """
    Run a deterministic RCA + investigation pass for an arbitrary incident
    (not necessarily the currently active one), without disturbing the
    runtime's own tick-driven ``_rca_result``/``_investigation_report``
    state.
    """
    runtime.rca_service.historical_incidents = list(runtime.historical_incidents)
    rca_result = runtime.rca_service.analyze(
        incident,
        step=runtime.step,
        drift_findings=runtime._drift_findings,
        quality_issues=runtime._quality_issues,
        feature_observations=runtime._recent_feature_observations,
        deployment_events=runtime._deployment_events,
    )
    ad_hoc_snapshot = SimpleNamespace(incident=incident, rca_result=rca_result)
    report = runtime.investigator.investigate(
        snapshot=ad_hoc_snapshot,
        historical_incidents=list(runtime.historical_incidents),
    )
    return {
        "incident": incident.to_dict(),
        "rca_result": rca_result.to_dict(),
        "investigation_report": None if report is None else report.to_dict(),
    }


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
            if self.path == "/api/investigation":
                snapshot = runtime.snapshot()
                self._send_json(
                    {
                        "investigation_report": None if snapshot.investigation_report is None else snapshot.investigation_report.to_dict(),
                        "historical_incidents": [incident.to_dict() for incident in snapshot.historical_incidents],
                    }
                )
                return
            if self.path == "/api/incidents":
                self._send_json(
                    {"incidents": [incident.to_dict() for incident in runtime.incident_engine.incidents]}
                )
                return
            if self.path.startswith("/api/incidents/"):
                incident_id = self.path[len("/api/incidents/") :].strip("/")
                if not incident_id:
                    self.send_error(404, "Not found")
                    return
                try:
                    incident = runtime.incident_engine.get_incident(incident_id)
                except KeyError:
                    self.send_error(404, f"Unknown incident_id: {incident_id}")
                    return
                self._send_json(incident.to_dict())
                return
            if self.path in {"/health", "/healthz"}:
                self._send_json({"status": "ok", "step": runtime.step})
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/api/incidents/") and self.path.endswith("/investigate"):
                incident_id = self.path[len("/api/incidents/") : -len("/investigate")].strip("/")
                incident = None
                if incident_id:
                    try:
                        incident = runtime.incident_engine.get_incident(incident_id)
                    except KeyError:
                        incident = None
                if incident is None:
                    self.send_error(404, f"Unknown incident_id: {incident_id}")
                    return
                self._send_json(_investigate_incident(runtime, incident))
                return

            if self.path == "/api/investigation":
                body = self._read_json_body()
                incident_id = body.get("incident_id") if isinstance(body, dict) else None
                incident = None
                if incident_id:
                    try:
                        incident = runtime.incident_engine.get_incident(incident_id)
                    except KeyError:
                        incident = None
                else:
                    incident = runtime.active_incident
                if incident is None:
                    self.send_error(
                        404,
                        "No active incident and no valid incident_id was provided.",
                    )
                    return
                self._send_json(_investigate_incident(runtime, incident))
                return

            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json_body(self) -> object:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}

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