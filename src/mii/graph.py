from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from math import exp
from typing import Any, Iterable

from .monitoring import DataQualityIssue, DriftFinding, FeatureObservation
from .models import DetectedAnomaly


@dataclass(slots=True)
class GraphNode:
    """A component in the dependency graph."""

    node_id: str
    label: str
    kind: str
    team: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "kind": self.kind,
            "team": self.team,
        }


@dataclass(slots=True)
class GraphEdge:
    """Directed dependency edge from upstream to downstream."""

    source: str
    target: str
    relationship: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
        }


@dataclass(slots=True)
class DeploymentEvent:
    """A deployment event that can be used as RCA evidence."""

    timestamp: datetime
    component: str
    version: str
    step: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "version": self.version,
            "step": self.step,
            "description": self.description,
        }


@dataclass(slots=True)
class RootCauseHypothesis:
    """A ranked RCA hypothesis with supporting evidence."""

    component: str
    label: str
    score: float
    confidence: float
    reasons: list[str]
    evidence_signals: list[str]
    downstream_effects: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "evidence_signals": list(self.evidence_signals),
            "downstream_effects": list(self.downstream_effects),
        }


@dataclass(slots=True)
class DependencyGraph:
    """Simple directed acyclic graph for system dependencies."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    _node_index: dict[str, GraphNode] = field(init=False, repr=False)
    _children: dict[str, list[str]] = field(init=False, repr=False)
    _parents: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._node_index = {node.node_id: node for node in self.nodes}
        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        for edge in self.edges:
            self._children[edge.source].append(edge.target)
            self._parents[edge.target].append(edge.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def node(self, node_id: str) -> GraphNode | None:
        return self._node_index.get(node_id)

    def children(self, node_id: str) -> list[str]:
        return list(self._children.get(node_id, []))

    def parents(self, node_id: str) -> list[str]:
        return list(self._parents.get(node_id, []))

    def descendants(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        queue = deque(self.children(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.children(current))
        return seen

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        queue = deque(self.parents(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.parents(current))
        return seen

    def path_distance(self, source: str, target: str) -> int | None:
        """Return shortest directed path distance from source to target."""

        queue = deque([(source, 0)])
        seen = {source}
        while queue:
            current, distance = queue.popleft()
            if current == target:
                return distance
            for child in self.children(current):
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, distance + 1))
        return None


class GraphRCAEngine:
    """Rank root cause hypotheses using dependency topology and evidence."""

    def __init__(
        self,
        graph: DependencyGraph,
        *,
        deployment_recency_window: int = 6,
        top_k: int = 3,
    ) -> None:
        self.graph = graph
        self.deployment_recency_window = deployment_recency_window
        self.top_k = top_k

    def analyze(
        self,
        *,
        step: int,
        anomalies: list[DetectedAnomaly],
        drift_findings: list[DriftFinding],
        quality_issues: list[DataQualityIssue],
        feature_observations: list[FeatureObservation],
        deployment_events: list[DeploymentEvent],
    ) -> list[RootCauseHypothesis]:
        signal_bucket: dict[str, list[str]] = defaultdict(list)
        score_bucket: dict[str, float] = defaultdict(float)
        direct_evidence: dict[str, list[str]] = defaultdict(list)

        for anomaly in anomalies:
            signal_bucket[anomaly.entity].append(f"{anomaly.metric}: {anomaly.explanation}")
            score_bucket[anomaly.entity] += _score_anomaly(anomaly)
            direct_evidence[anomaly.entity].append(anomaly.metric)

        for finding in drift_findings:
            signal_bucket["feature-service"].append(f"drift:{finding.feature_name} psi={finding.psi:.3f}")
            score_bucket["feature-service"] += min(0.28, finding.psi * 0.42)
            direct_evidence["feature-service"].append(f"drift:{finding.feature_name}")

        for issue in quality_issues:
            signal_bucket["feature-service"].append(f"dq:{issue.feature_name} {issue.issue_type}")
            score_bucket["feature-service"] += _score_quality_issue(issue)
            direct_evidence["feature-service"].append(f"dq:{issue.feature_name}")

        for observation in feature_observations:
            if observation.value is None:
                continue
            signal_bucket[observation.entity].append(f"feature:{observation.feature_name}")
            score_bucket[observation.entity] += 0.01

        active_deployments = [event for event in deployment_events if step - event.step <= self.deployment_recency_window]
        for deployment in active_deployments:
            signal_bucket[deployment.component].append(f"deploy:{deployment.version}")
            score_bucket[deployment.component] += 0.34
            direct_evidence[deployment.component].append(f"deploy:{deployment.version}")

        for node in self.graph.nodes:
            descendants = self.graph.descendants(node.node_id)
            impacted_descendants = [desc for desc in descendants if desc in score_bucket]
            if impacted_descendants:
                score_bucket[node.node_id] += min(0.22, 0.05 * len(impacted_descendants))
                for descendant in impacted_descendants:
                    distance = self.graph.path_distance(node.node_id, descendant) or 1
                    score_bucket[node.node_id] += max(0.0, 0.08 / distance)

            if node.node_id in score_bucket and self.graph.children(node.node_id):
                score_bucket[node.node_id] += 0.05

            if node.node_id in score_bucket:
                ancestors = self.graph.ancestors(node.node_id)
                score_bucket[node.node_id] += min(0.08, 0.02 * len(ancestors))

        hypotheses: list[RootCauseHypothesis] = []
        for node_id, score in sorted(score_bucket.items(), key=lambda item: item[1], reverse=True):
            node = self.graph.node(node_id)
            if node is None:
                continue
            hypotheses.append(
                RootCauseHypothesis(
                    component=node.node_id,
                    label=node.label,
                    score=round(min(score, 1.0), 3),
                    confidence=round(_confidence_from_score(score), 2),
                    reasons=_reasons_for_component(
                        node=node,
                        score=score,
                        signals=signal_bucket.get(node_id, []),
                        deployment_events=active_deployments,
                        graph=self.graph,
                    ),
                    evidence_signals=sorted(set(direct_evidence.get(node_id, []))),
                    downstream_effects=sorted(self.graph.descendants(node_id)),
                )
            )

        if not hypotheses:
            return [
                RootCauseHypothesis(
                    component="unknown",
                    label="Insufficient evidence",
                    score=0.0,
                    confidence=0.0,
                    reasons=["No anomalies, drift, quality issues, or deployment evidence available."],
                    evidence_signals=[],
                    downstream_effects=[],
                )
            ]

        return hypotheses[: self.top_k]


def build_default_dependency_graph() -> DependencyGraph:
    return DependencyGraph(
        nodes=[
            GraphNode("data-pipeline", "Data Pipeline", "pipeline", "data-platform"),
            GraphNode("feature-service", "Feature Service", "service", "ml-platform"),
            GraphNode("model-serving", "Model Serving", "service", "ml-platform"),
            GraphNode("api-gateway", "API Gateway", "service", "platform"),
        ],
        edges=[
            GraphEdge("data-pipeline", "feature-service", "feeds_features"),
            GraphEdge("feature-service", "model-serving", "provides_features"),
            GraphEdge("model-serving", "api-gateway", "serves_predictions"),
            GraphEdge("feature-service", "api-gateway", "feeds_request_context"),
        ],
    )


def _score_anomaly(anomaly: DetectedAnomaly) -> float:
    severity_weight = {
        "critical": 0.36,
        "high": 0.28,
        "medium": 0.20,
        "low": 0.12,
    }.get(anomaly.severity, 0.18)
    metric_weight = {
        "prediction_latency_ms": 0.20,
        "feature_missing_rate": 0.22,
        "error_rate": 0.18,
        "prediction_ctr": 0.18,
        "traffic_rps": 0.10,
    }.get(anomaly.metric, 0.10)
    return severity_weight + metric_weight


def _score_quality_issue(issue: DataQualityIssue) -> float:
    issue_weight = {
        "missing_rate": 0.24,
        "out_of_range": 0.20,
        "duplicate_request": 0.10,
    }.get(issue.issue_type, 0.08)
    return issue_weight + min(0.18, issue.rate * 0.34)


def _confidence_from_score(score: float) -> float:
    return min(0.99, 0.34 + 0.72 * (1.0 - exp(-score)))


def _reasons_for_component(
    *,
    node: GraphNode,
    score: float,
    signals: list[str],
    deployment_events: list[DeploymentEvent],
    graph: DependencyGraph,
) -> list[str]:
    reasons: list[str] = []
    deployment = next((event for event in deployment_events if event.component == node.node_id), None)
    if deployment is not None:
        reasons.append(f"Deployment {deployment.version} landed on {deployment.component} at step {deployment.step}.")

    if signals:
        reasons.append(f"Direct evidence observed on {node.label}: {', '.join(signals[:3])}.")

    descendants = sorted(graph.descendants(node.node_id))
    if descendants:
        reasons.append(f"{node.label} sits upstream of {', '.join(descendants)} in the dependency graph.")

    if score >= 0.8:
        reasons.append("Strong combined evidence from telemetry, drift, quality, and graph topology.")
    elif score >= 0.5:
        reasons.append("Multiple signals support this hypothesis, though some uncertainty remains.")
    else:
        reasons.append("Weak supporting evidence; included as a lower-ranked candidate.")

    return reasons
