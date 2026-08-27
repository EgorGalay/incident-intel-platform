from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mii.detectors import EWMADeviationDetector, RollingZScoreDetector
from mii.models import MetricSample


class DetectorTests(unittest.TestCase):
    def test_rolling_zscore_detects_spike(self) -> None:
        detector = RollingZScoreDetector(window_size=8, threshold=3.0, min_samples=6)
        timestamp = datetime.now(timezone.utc)

        for _ in range(8):
            self.assertIsNone(
                detector.observe(
                    MetricSample(timestamp, "prediction_latency_ms", "feature-service", 100.0, "ms")
                )
            )

        anomaly = detector.observe(MetricSample(timestamp, "prediction_latency_ms", "feature-service", 240.0, "ms"))
        self.assertIsNotNone(anomaly)
        self.assertGreater(anomaly.z_score, 3.0)

    def test_ewma_detector_detects_shift(self) -> None:
        detector = EWMADeviationDetector(alpha=0.3, threshold=2.5, warmup=5)
        timestamp = datetime.now(timezone.utc)

        for _ in range(6):
            self.assertIsNone(detector.observe(MetricSample(timestamp, "feature_missing_rate", "feature-service", 0.02, "ratio")))

        anomaly = detector.observe(MetricSample(timestamp, "feature_missing_rate", "feature-service", 0.31, "ratio"))
        self.assertIsNotNone(anomaly)
        self.assertGreater(anomaly.z_score, 2.5)


if __name__ == "__main__":
    unittest.main()
