from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from mii.investigation import IncidentInvestigator
from mii.state import PhaseOneRuntime
from mii.synthetic import FeatureProfile, FaultProfile, SyntheticWorkload


class PhaseFourInvestigationTests(unittest.TestCase):
    def test_runtime_generates_grounded_investigation_report(self) -> None:
        runtime = PhaseOneRuntime(
            workload=SyntheticWorkload(
                seed=17,
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
        for _ in range(28):
            snapshot = runtime.tick()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.investigation_report)
        assert snapshot.investigation_report is not None
        self.assertIn(snapshot.investigation_report.mode, {"ml-sufficient", "local-fallback", "llm"})
        self.assertGreater(len(snapshot.investigation_report.evidence), 0)
        self.assertGreater(len(snapshot.historical_incidents), 0)
        self.assertTrue(snapshot.investigation_report.tool_trace)
        self.assertGreaterEqual(snapshot.investigation_report.usage.input_tokens, 1)

    def test_investigator_can_run_without_llm(self) -> None:
        runtime = PhaseOneRuntime()
        for _ in range(24):
            snapshot = runtime.tick()

        investigator = IncidentInvestigator(llm_confidence_threshold=0.0)
        report = investigator.investigate(
            snapshot=snapshot,
            historical_incidents=list(snapshot.historical_incidents),
        )

        self.assertEqual(report.mode, "ml-sufficient")
        self.assertIn("Grounded investigation", report.summary)
        self.assertGreater(len(report.recommended_actions), 0)
        self.assertGreater(len(report.evidence), 0)

    @patch("mii.investigation.urlopen")
    def test_investigator_uses_ollama_when_available(self, mock_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"response": "Local Ollama summary."}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        mock_urlopen.return_value = response

        runtime = PhaseOneRuntime()
        for _ in range(24):
            snapshot = runtime.tick()

        investigator = IncidentInvestigator(llm_provider="ollama", ollama_model="llama3.1")
        report = investigator.investigate(
            snapshot=snapshot,
            historical_incidents=list(snapshot.historical_incidents),
        )

        self.assertEqual(report.mode, "llm")
        self.assertEqual(report.usage.provider, "ollama")
        self.assertEqual(report.usage.model, "llama3.1")
        self.assertIn("Local Ollama summary", report.summary)
        self.assertEqual(report.skipped_reason, None)


if __name__ == "__main__":
    unittest.main()
