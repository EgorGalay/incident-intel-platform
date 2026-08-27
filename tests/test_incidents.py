from __future__ import annotations

import unittest

from mii.state import PhaseOneRuntime
from mii.synthetic import FaultProfile, SyntheticWorkload


class IncidentTests(unittest.TestCase):
    def test_runtime_creates_incident_after_correlated_anomalies(self) -> None:
        runtime = PhaseOneRuntime(
            workload=SyntheticWorkload(seed=4, fault_profile=FaultProfile(trigger_step=9)),
        )

        snapshot = None
        for _ in range(25):
            snapshot = runtime.tick()
            if snapshot.incident is not None:
                break

        self.assertIsNotNone(snapshot)
        self.assertIsNotNone(snapshot.incident)
        assert snapshot.incident is not None
        self.assertEqual(snapshot.incident.status, "OPEN")
        self.assertIn("Feature service", snapshot.incident.title)
        self.assertGreaterEqual(len(snapshot.incident.observed_anomalies), 3)


if __name__ == "__main__":
    unittest.main()
