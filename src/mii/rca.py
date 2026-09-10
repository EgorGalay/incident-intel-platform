from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .graph import (
    DependencyGraph,
    DeploymentEvent,
    GraphRCAEngine,
    RootCauseHypothesis,
)
from .models import DetectedAnomaly, Incident


# ---------------------------------------------------------------------------
# RCA result models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RCAResult:
    """
    Structured result of deterministic root-cause analysis.

    This object is deliberately independent from the LLM investigation layer.
    It represents what the platform can conclude from telemetry, topology,
    deployments, drift, and data-quality evidence alone.
    """

    incident_id: str | None
    root_cause: str | None
    root_cause_label: str | None
    confidence: float
    score: float
    hypotheses: list[RootCauseHypothesis] = field(
        default_factory=list
    )
    evidence: list[str] = field(
        default_factory=list
    )
    downstream_effects: list[str] = field(
        default_factory=list
    )
    causal_chain: list[str] = field(
        default_factory=list
    )
    historical_matches: list[Any] = field(
        default_factory=list
    )

    @property
    def verdict(self) -> str:
        """Human-readable RCA verdict."""
        if not self.root_cause:
            return "insufficient_evidence"

        if self.confidence >= 0.80:
            return "high_confidence_root_cause"

        if self.confidence >= 0.60:
            return "probable_root_cause"

        if self.confidence >= 0.40:
            return "weak_root_cause"

        return "insufficient_evidence"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "incident_id": self.incident_id,
            "root_cause": self.root_cause,
            "root_cause_label": self.root_cause_label,
            "confidence": self.confidence,
            "score": self.score,
            "verdict": self.verdict,
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in self.hypotheses
            ],
            "evidence": list(self.evidence),
            "downstream_effects": list(
                self.downstream_effects
            ),
            "causal_chain": list(
                self.causal_chain
            ),
            "historical_matches": [
                _serialize_historical_match(match)
                for match in self.historical_matches
            ],
        }


# ---------------------------------------------------------------------------
# RCA service
# ---------------------------------------------------------------------------


class RCAService:
    """
    Application-level deterministic RCA service.

    Responsibilities
    ----------------
    - collect incident evidence
    - invoke graph-aware RCA
    - rank hypotheses
    - find historical matches
    - explain dependency propagation
    - produce an InvestigationService-friendly evidence bundle

    It intentionally does NOT:
    - call an LLM
    - make HTTP requests
    - mutate incidents
    - execute remediation
    """

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        historical_incidents: Iterable[Any] = (),
        *,
        top_k: int = 3,
        historical_limit: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if historical_limit < 0:
            raise ValueError(
                "historical_limit cannot be negative."
            )

        self.dependency_graph = dependency_graph

        self.historical_incidents = list(
            historical_incidents
        )

        self.historical_limit = historical_limit

        self.engine = GraphRCAEngine(
            dependency_graph,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        incident: Incident,
        *,
        step: int = 0,
        drift_findings: Sequence[Any] = (),
        quality_issues: Sequence[Any] = (),
        feature_observations: Sequence[Any] = (),
        deployment_events: Sequence[DeploymentEvent] = (),
    ) -> RCAResult:
        """
        Perform deterministic RCA for an incident.
        """
        anomalies = list(
            incident.anomalies
        )

        hypotheses = self.engine.analyze(
            step=step,
            anomalies=anomalies,
            drift_findings=list(
                drift_findings
            ),
            quality_issues=list(
                quality_issues
            ),
            feature_observations=list(
                feature_observations
            ),
            deployment_events=list(
                deployment_events
            ),
        )

        ranked = self.rank_root_causes(
            hypotheses
        )

        if not ranked:
            return RCAResult(
                incident_id=incident.incident_id,
                root_cause=None,
                root_cause_label=None,
                confidence=0.0,
                score=0.0,
            )

        best = ranked[0]

        historical_matches = self.find_historical_matches(
            incident
        )

        causal_chain = self.explain_chain(
            best.component,
            affected_components=incident.affected_components,
        )

        evidence = self._build_evidence(
            incident=incident,
            hypothesis=best,
            deployment_events=deployment_events,
            drift_findings=drift_findings,
            quality_issues=quality_issues,
        )

        return RCAResult(
            incident_id=incident.incident_id,
            root_cause=best.component,
            root_cause_label=best.label,
            confidence=best.confidence,
            score=best.score,
            hypotheses=ranked,
            evidence=evidence,
            downstream_effects=list(
                best.downstream_effects
            ),
            causal_chain=causal_chain,
            historical_matches=historical_matches,
        )

    # ------------------------------------------------------------------
    # Hypothesis ranking
    # ------------------------------------------------------------------

    def rank_root_causes(
        self,
        hypotheses: Iterable[RootCauseHypothesis],
    ) -> list[RootCauseHypothesis]:
        """
        Rank hypotheses by combined score and confidence.

        ``GraphRCAEngine`` already produces a meaningful score. The small
        confidence tie-breaker prevents equally-scored hypotheses from being
        ordered arbitrarily.
        """
        return sorted(
            hypotheses,
            key=lambda hypothesis: (
                hypothesis.score,
                hypothesis.confidence,
                len(hypothesis.evidence_signals),
            ),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Historical matching
    # ------------------------------------------------------------------

    def find_historical_matches(
        self,
        incident: Incident,
        *,
        limit: int | None = None,
    ) -> list[Any]:
        """
        Find deterministic historical matches.

        Matching uses:
        - component overlap
        - metric overlap
        - severity similarity
        - root-cause/component similarity where available

        This is intentionally lightweight. It gives the future LLM a
        grounded historical context without requiring embeddings or a vector
        database.
        """
        requested_limit = (
            self.historical_limit
            if limit is None
            else limit
        )

        if requested_limit <= 0:
            return []

        current_metrics = {
            anomaly.metric
            for anomaly in incident.anomalies
        }

        current_components = {
            incident.component,
            *incident.affected_components,
        }

        scored: list[
            tuple[float, Any]
        ] = []

        for historical in self.historical_incidents:
            score = self._historical_similarity(
                incident=incident,
                historical=historical,
                current_metrics=current_metrics,
                current_components=current_components,
            )

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    historical,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            item[1]
            for item in scored[:requested_limit]
        ]

    def _historical_similarity(
        self,
        *,
        incident: Incident,
        historical: Any,
        current_metrics: set[str],
        current_components: set[str],
    ) -> float:
        historical_component = getattr(
            historical,
            "component",
            None,
        )

        historical_components = set()

        if historical_component:
            historical_components.add(
                str(historical_component)
            )

        historical_components.update(
            str(component)
            for component in getattr(
                historical,
                "affected_components",
                [],
            )
        )

        historical_anomalies = getattr(
            historical,
            "anomalies",
            [],
        )

        historical_metrics = {
            str(
                getattr(
                    anomaly,
                    "metric",
                    "",
                )
            )
            for anomaly in historical_anomalies
            if getattr(
                anomaly,
                "metric",
                None,
            )
        }

        component_score = _jaccard(
            current_components,
            historical_components,
        )

        metric_score = _jaccard(
            current_metrics,
            historical_metrics,
        )

        severity_score = 0.0

        historical_severity = getattr(
            historical,
            "severity",
            None,
        )

        if historical_severity == incident.severity:
            severity_score = 1.0
        elif historical_severity:
            severity_score = 0.4

        # Historical objects in the existing project sometimes expose
        # ``root_cause`` instead of a component field.
        historical_root_cause = getattr(
            historical,
            "root_cause",
            None,
        )

        root_cause_score = (
            0.5
            if historical_root_cause
            and historical_root_cause
            in current_components
            else 0.0
        )

        return (
            0.40 * component_score
            + 0.35 * metric_score
            + 0.15 * severity_score
            + 0.10 * root_cause_score
        )

    # ------------------------------------------------------------------
    # Dependency explanation
    # ------------------------------------------------------------------

    def explain_chain(
        self,
        root_cause: str,
        *,
        affected_components: Iterable[str] = (),
    ) -> list[str]:
        """
        Explain how the suspected root cause can propagate downstream.
        """
        affected = set(
            affected_components
        )

        if root_cause not in self.dependency_graph._node_index:
            return [
                f"Unknown root-cause component: {root_cause}"
            ]

        chains: list[str] = []

        if not affected:
            descendants = sorted(
                self.dependency_graph.descendants(
                    root_cause
                )
            )

            for descendant in descendants:
                path = self.dependency_graph.path(
                    root_cause,
                    descendant,
                )

                if path:
                    chains.append(
                        " -> ".join(path)
                    )

            return chains

        for component in sorted(affected):
            if component == root_cause:
                chains.append(root_cause)
                continue

            path = self.dependency_graph.path(
                root_cause,
                component,
            )

            if path:
                chains.append(
                    " -> ".join(path)
                )

        if not chains:
            chains.append(
                f"No dependency path from "
                f"{root_cause} to the affected "
                f"components was found."
            )

        return chains

    # ------------------------------------------------------------------
    # Evidence bundle
    # ------------------------------------------------------------------

    def build_evidence_bundle(
        self,
        incident: Incident,
        result: RCAResult,
    ) -> dict[str, Any]:
        """
        Build the compact, grounded context consumed by investigation.py.

        The bundle intentionally contains structured evidence rather than
        free-form model output.
        """
        return {
            "incident": incident.to_dict(),
            "rca": result.to_dict(),
            "dependency_graph": self.dependency_graph.to_dict(),
            "historical_matches": [
                _serialize_historical_match(match)
                for match in result.historical_matches
            ],
        }

    # ------------------------------------------------------------------
    # Evidence formatting
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        *,
        incident: Incident,
        hypothesis: RootCauseHypothesis,
        deployment_events: Sequence[DeploymentEvent],
        drift_findings: Sequence[Any],
        quality_issues: Sequence[Any],
    ) -> list[str]:
        evidence: list[str] = []

        for anomaly in incident.anomalies:
            evidence.append(
                f"{anomaly.timestamp.isoformat()} "
                f"{anomaly.entity}/{anomaly.metric}: "
                f"value={anomaly.value:.4f}, "
                f"baseline={anomaly.baseline:.4f}, "
                f"z={anomaly.z_score:.2f}, "
                f"severity={anomaly.severity}."
            )

        for signal in hypothesis.evidence_signals:
            evidence.append(
                f"Supporting signal: {signal}."
            )

        for deployment in deployment_events:
            if deployment.component == hypothesis.component:
                evidence.append(
                    f"Recent deployment on "
                    f"{deployment.component}: "
                    f"{deployment.version} "
                    f"at step {deployment.step}."
                )

        for finding in drift_findings:
            feature_name = getattr(
                finding,
                "feature_name",
                "unknown",
            )

            psi = getattr(
                finding,
                "psi",
                None,
            )

            if psi is not None:
                evidence.append(
                    f"Feature drift detected for "
                    f"{feature_name}: PSI={float(psi):.3f}."
                )

        for issue in quality_issues:
            feature_name = getattr(
                issue,
                "feature_name",
                "unknown",
            )

            issue_type = getattr(
                issue,
                "issue_type",
                "unknown",
            )

            evidence.append(
                f"Data-quality issue for "
                f"{feature_name}: {issue_type}."
            )

        return _deduplicate_preserving_order(
            evidence
        )


# ---------------------------------------------------------------------------
# Backward-compatible facade
# ---------------------------------------------------------------------------


class RootCauseAnalyzer:
    """
    Compatibility facade.

    Existing code can continue importing ``RootCauseAnalyzer`` from either
    ``mii.graph`` or ``mii.rca``.

    New application code should prefer ``RCAService`` because it returns a
    typed RCAResult.
    """

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        historical_incidents: Iterable[Any] = (),
        *,
        top_k: int = 3,
    ) -> None:
        self.service = RCAService(
            dependency_graph,
            historical_incidents,
            top_k=top_k,
        )

    def analyze(
        self,
        incident: Incident,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the legacy dictionary RCA representation."""
        return self.service.analyze(
            incident,
            **kwargs,
        ).to_dict()

    def rank_root_causes(
        self,
        hypotheses: Iterable[RootCauseHypothesis],
    ) -> list[RootCauseHypothesis]:
        return self.service.rank_root_causes(
            hypotheses
        )

    def find_historical_matches(
        self,
        incident: Incident,
        *,
        limit: int = 5,
    ) -> list[Any]:
        return self.service.find_historical_matches(
            incident,
            limit=limit,
        )

    def explain_chain(
        self,
        root_cause: str,
        affected_component: str | None = None,
    ) -> list[str]:
        affected = (
            [affected_component]
            if affected_component is not None
            else []
        )

        return self.service.explain_chain(
            root_cause,
            affected_components=affected,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _jaccard(
    left: set[str],
    right: set[str],
) -> float:
    if not left and not right:
        return 1.0

    union = left | right

    if not union:
        return 0.0

    return len(left & right) / len(union)


def _deduplicate_preserving_order(
    values: Iterable[str],
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _serialize_historical_match(
    historical: Any,
) -> dict[str, Any]:
    """
    Convert arbitrary historical-incident representations into a safe,
    compact dictionary.
    """
    if isinstance(historical, dict):
        return {
            key: value
            for key, value in historical.items()
            if key not in {
                "api_key",
                "token",
                "authorization",
                "secret",
            }
        }

    result: dict[str, Any] = {}

    for attribute in (
        "incident_id",
        "title",
        "severity",
        "status",
        "component",
        "root_cause",
        "confidence",
        "summary",
    ):
        if hasattr(historical, attribute):
            result[attribute] = getattr(
                historical,
                attribute,
            )

    # Preserve the most useful metric information.
    anomalies = getattr(
        historical,
        "anomalies",
        [],
    )

    if anomalies:
        result["metrics"] = sorted(
            {
                str(
                    getattr(
                        anomaly,
                        "metric",
                        "",
                    )
                )
                for anomaly in anomalies
                if getattr(
                    anomaly,
                    "metric",
                    None,
                )
            }
        )

    return result


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "RCAResult",
    "RCAService",
    "RootCauseAnalyzer",
]

