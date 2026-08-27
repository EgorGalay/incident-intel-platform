from __future__ import annotations

import unittest

from mii.synthetic import FaultProfile, SyntheticWorkload


class SyntheticWorkloadTests(unittest.TestCase):
    def test_fault_injection_changes_key_metrics(self) -> None:
        workload = SyntheticWorkload(seed=1, fault_profile=FaultProfile(trigger_step=2))

        before = workload.next_frame()
        after = workload.next_frame()

        before_map = {sample.metric: sample.value for sample in before}
        after_map = {sample.metric: sample.value for sample in after}

        self.assertLess(after_map["prediction_ctr"], before_map["prediction_ctr"])
        self.assertGreater(after_map["prediction_latency_ms"], before_map["prediction_latency_ms"])
        self.assertGreater(after_map["feature_missing_rate"], before_map["feature_missing_rate"])
        self.assertGreater(after_map["error_rate"], before_map["error_rate"])


if __name__ == "__main__":
    unittest.main()
