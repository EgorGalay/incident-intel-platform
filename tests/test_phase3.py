from __future__ import annotations

import unittest

from mii.graph import build_default_dependency_graph
from mii.state import PhaseOneRuntime
from mii.synthetic import FeatureProfile, FaultProfile, SyntheticWorkload


class PhaseThreeRcaTests(unittest.TestCase):
    def test_dependency_graph_contains_expected_path(self) -> None:
        graph = build_default_dependency_graph()

        self.assertIn("model-serving", graph.descendants("feature-service"))
        self.assertIn("api-gateway", graph.descendants("feature-service"))
        self.assertEqual(graph.path_distance("data-pipeline", "api-gateway"), 2)

    def test_runtime_ranks_feature_service_as_primary_hypothesis(self) -> None:
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

        snapshot = None
        for _ in range(26):
            snapshot = runtime.tick()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.deployment_events)
        self.assertTrue(snapshot.root_cause_hypotheses)
        self.assertEqual(snapshot.root_cause_hypotheses[0].component, "feature-service")
        self.assertGreater(snapshot.root_cause_hypotheses[0].confidence, 0.6)
        self.assertIn("model-serving", snapshot.root_cause_hypotheses[0].downstream_effects)
        self.assertIn("api-gateway", snapshot.root_cause_hypotheses[0].downstream_effects)


if __name__ == "__main__":
    unittest.main()
