from __future__ import annotations

import json
import threading
import unittest
import time
from urllib.request import urlopen

from mii.server import create_server
from mii.state import PhaseOneRuntime


class ServerTests(unittest.TestCase):
    def test_health_and_state_endpoints(self) -> None:
        runtime = PhaseOneRuntime()
        runtime.tick()

        server = create_server(runtime, "127.0.0.1", 0)
        port = server.server_address[1]
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        time.sleep(0.05)

        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")

            with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as response:
                state = json.loads(response.read().decode("utf-8"))
            self.assertIn("latest_metrics", state)
            self.assertIn("recent_anomalies", state)
            self.assertIn("dependency_graph", state)
            self.assertIn("root_cause_hypotheses", state)

            with urlopen(f"http://127.0.0.1:{port}/api/graph", timeout=5) as response:
                graph = json.loads(response.read().decode("utf-8"))
            self.assertIn("nodes", graph)
            self.assertIn("edges", graph)

            with urlopen(f"http://127.0.0.1:{port}/api/rca", timeout=5) as response:
                rca = json.loads(response.read().decode("utf-8"))
            self.assertIn("root_cause_hypotheses", rca)

            with urlopen(f"http://127.0.0.1:{port}/api/investigation", timeout=5) as response:
                investigation = json.loads(response.read().decode("utf-8"))
            self.assertIn("historical_incidents", investigation)
            self.assertIn("investigation_report", investigation)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
