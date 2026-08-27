from __future__ import annotations

import unittest

from mii.dashboard import render_dashboard
from mii.state import PhaseOneRuntime
from mii.synthetic import FaultProfile, SyntheticWorkload


class DashboardTests(unittest.TestCase):
    def test_dashboard_contains_phase_one_sections(self) -> None:
        runtime = PhaseOneRuntime(
            workload=SyntheticWorkload(seed=9, fault_profile=FaultProfile(trigger_step=4)),
        )
        for _ in range(10):
            snapshot = runtime.tick()

        html = render_dashboard(snapshot)

        self.assertIn("Phase 1 Live Dashboard", html)
        self.assertIn("/api/state", html)
        self.assertIn("Latest Metrics", html)


if __name__ == "__main__":
    unittest.main()
