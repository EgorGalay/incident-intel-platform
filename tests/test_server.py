from __future__ import annotations

import json
import threading
import unittest
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mii.server import create_server
from mii.state import PhaseOneRuntime
from mii.synthetic import FeatureProfile, FaultProfile, SyntheticWorkload


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

            with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
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

    def test_incident_endpoints_and_investigate_action(self) -> None:
        runtime = PhaseOneRuntime(
            workload=SyntheticWorkload(
                seed=13,
                fault_profile=FaultProfile(trigger_step=10),
                feature_profile=FeatureProfile(
                    drift_trigger_step=4,
                    quality_trigger_step=6,
                    deployment_trigger_step=8,
                    user_age_shift=15.0,
                    missing_probability=0.5,
                ),
            )
        )
        for _ in range(26):
            runtime.tick()
        incident_id = runtime.active_incident.incident_id

        server = create_server(runtime, "127.0.0.1", 0)
        port = server.server_address[1]
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        time.sleep(0.05)

        try:
            with urlopen(f"http://127.0.0.1:{port}/api/incidents", timeout=5) as response:
                incidents = json.loads(response.read().decode("utf-8"))
            self.assertIn("incidents", incidents)
            self.assertTrue(any(item["incident_id"] == incident_id for item in incidents["incidents"]))

            with urlopen(f"http://127.0.0.1:{port}/api/incidents/{incident_id}", timeout=5) as response:
                incident = json.loads(response.read().decode("utf-8"))
            self.assertEqual(incident["incident_id"], incident_id)

            with self.assertRaises(HTTPError) as ctx:
                urlopen(f"http://127.0.0.1:{port}/api/incidents/does-not-exist", timeout=5)
            self.assertEqual(ctx.exception.code, 404)

            request = Request(
                f"http://127.0.0.1:{port}/api/incidents/{incident_id}/investigate",
                data=b"",
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(result["incident"]["incident_id"], incident_id)
            self.assertEqual(result["rca_result"]["root_cause"], "feature-service")
            self.assertIn("investigation_report", result)

            request = Request(
                f"http://127.0.0.1:{port}/api/investigation",
                data=json.dumps({"incident_id": incident_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(result["incident"]["incident_id"], incident_id)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()