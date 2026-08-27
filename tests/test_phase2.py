from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mii.monitoring import DataQualityMonitor, FeatureObservation, DriftMonitor
from mii.state import PhaseOneRuntime
from mii.synthetic import FeatureProfile, SyntheticWorkload


class PhaseTwoMonitoringTests(unittest.TestCase):
    def test_drift_monitor_flags_distribution_shift(self) -> None:
        monitor = DriftMonitor(baseline_size=12, window_size=12, psi_threshold=0.2)
        timestamp = datetime.now(timezone.utc)

        result = None
        for idx in range(12):
            result = monitor.observe(
                FeatureObservation(timestamp, f"req-{idx:05d}", "user_age", "feature-service", 33.0 + (idx % 2))
            )
            self.assertIsNone(result)

        for idx in range(12, 28):
            result = monitor.observe(
                FeatureObservation(timestamp, f"req-{idx:05d}", "user_age", "feature-service", 51.0 + (idx % 3))
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.psi, 0.2)

    def test_data_quality_monitor_flags_missing_rate(self) -> None:
        monitor = DataQualityMonitor(window_size=10, missing_rate_threshold=0.2)
        timestamp = datetime.now(timezone.utc)

        result = None
        for idx in range(10):
            value = None if idx >= 7 else 4.0
            result = monitor.observe(
                FeatureObservation(timestamp, f"req-{idx:05d}", "user_age", "feature-service", value)
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.issue_type, "missing_rate")
        self.assertGreaterEqual(result.rate, 0.2)

    def test_runtime_surfaces_phase_two_signals(self) -> None:
        runtime = PhaseOneRuntime(
            workload=SyntheticWorkload(
                seed=11,
                feature_profile=FeatureProfile(
                    drift_trigger_step=2,
                    quality_trigger_step=2,
                    user_age_shift=16.0,
                    missing_probability=0.5,
                ),
            )
        )

        snapshot = None
        for _ in range(32):
            snapshot = runtime.tick()

        assert snapshot is not None
        self.assertTrue(snapshot.drift_findings)
        self.assertTrue(snapshot.quality_issues)


if __name__ == "__main__":
    unittest.main()
