from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp
from typing import Any, Iterable

from .models import DetectedAnomaly


# ---------------------------------------------------------------------------
# Canonical dependency graph models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DependencyNode:
    """
    Canonical dependency-graph node.

    ``name`` is the stable component identifier.

    ``component_type`` examples:
        service
        pipeline
        model
        database
        gateway
        external

    ``metadata`` contains optional UI/RCA information.
    """

    name: str
    component_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component_type": self.component_type,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DependencyEdge:
    """
    Directed dependency relation.

    The direction is:

        source -> target

    and represents an upstream-to-downstream relationship.
    """

    source: str
    target: str
    relation: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("DependencyEdge.source cannot be empty.")

        if not self.target:
            raise ValueError("DependencyEdge.target cannot be empty.")

        if not self.relation:
            raise ValueError("DependencyEdge.relation cannot be empty.")

        if self.weight < 0:
            raise ValueError("DependencyEdge.weight cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }


# ---------------------------------------------------------------------------
# Backward-compatible graph models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphNode:
    """
    Compatibility representation used by the original Phase 1-4 code.

    New code should use ``DependencyNode``.
    """

    node_id: str
    label: str
    kind: str
    team: str = ""

    def to_dependency_node(self) -> DependencyNode:
        return DependencyNode(
            name=self.node_id,
            component_type=self.kind,
            metadata={
                "label": self.label,
                "team": self.team,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "kind": self.kind,
            "team": self.team,
        }


@dataclass(slots=True)
class GraphEdge:
    """
    Compatibility representation used by the original Phase 1-4 code.

    New code should use ``DependencyEdge``.
    """

    source: str
    target: str
    relationship: str
    weight: float = 1.0

    def to_dependency_edge(self) -> DependencyEdge:
        return DependencyEdge(
            source=self.source,
            target=self.target,
            relation=self.relationship,
            weight=self.weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "weight": self.weight,
        }


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------


@dataclass
class DependencyGraph:
    """
    Directed dependency graph.

    The graph is intentionally dependency-library-free. For this project's
    scale, adjacency dictionaries plus BFS provide everything required for
    topology-aware RCA.

    The graph supports both canonical ``DependencyNode``/
    ``DependencyEdge`` objects and legacy ``GraphNode``/``GraphEdge`` objects.
    """

    nodes: list[DependencyNode | GraphNode] = field(
        default_factory=list
    )

    edges: list[DependencyEdge | GraphEdge] = field(
        default_factory=list
    )

    _node_index: dict[str, DependencyNode] = field(
        init=False,
        repr=False,
    )

    _children: dict[str, list[str]] = field(
        init=False,
        repr=False,
    )

    _parents: dict[str, list[str]] = field(
        init=False,
        repr=False,
    )

    _edge_index: dict[tuple[str, str], DependencyEdge] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        original_nodes = list(self.nodes)
        original_edges = list(self.edges)

        self.nodes = []
        self.edges = []

        for node in original_nodes:
            if isinstance(node, DependencyNode):
                normalized = node
            elif isinstance(node, GraphNode):
                normalized = node.to_dependency_node()
            else:
                raise TypeError(
                    "DependencyGraph nodes must be "
                    "DependencyNode or GraphNode."
                )

            self.nodes.append(normalized)

        for edge in original_edges:
            if isinstance(edge, DependencyEdge):
                normalized_edge = edge
            elif isinstance(edge, GraphEdge):
                normalized_edge = edge.to_dependency_edge()
            else:
                raise TypeError(
                    "DependencyGraph edges must be "
                    "DependencyEdge or GraphEdge."
                )

            self.edges.append(normalized_edge)

        self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        self._node_index = {
            node.name: node
            for node in self.nodes
        }

        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        self._edge_index = {}

        for edge in self.edges:
            self._children[edge.source].append(edge.target)
            self._parents[edge.target].append(edge.source)
            self._edge_index[(edge.source, edge.target)] = edge

    def add_node(
        self,
        node: DependencyNode | GraphNode,
        *,
        replace: bool = False,
    ) -> DependencyNode:
        """
        Add a node.

        By default duplicate node names are rejected.
        """
        if isinstance(node, GraphNode):
            normalized = node.to_dependency_node()
        elif isinstance(node, DependencyNode):
            normalized = node
        else:
            raise TypeError(
                "node must be DependencyNode or GraphNode."
            )

        if normalized.name in self._node_index:
            if not replace:
                return self._node_index[normalized.name]

            self.nodes = [
                existing
                for existing in self.nodes
                if existing.name != normalized.name
            ]

        self.nodes.append(normalized)
        self._rebuild_indexes()

        return normalized

    def add_edge(
            self,
            edge: DependencyEdge | GraphEdge,
    ) -> DependencyEdge:
        """Add a dependency edge."""
        if isinstance(edge, GraphEdge):
            normalized = edge.to_dependency_edge()
        elif isinstance(edge, DependencyEdge):
            normalized = edge
        else:
            raise TypeError(
                "edge must be DependencyEdge or GraphEdge."
            )

        if normalized.source not in self._node_index:
            raise ValueError(
                f"Unknown dependency source: {normalized.source!r}"
            )

        if normalized.target not in self._node_index:
            raise ValueError(
                f"Unknown dependency target: {normalized.target!r}"
            )

        key = (
            normalized.source,
            normalized.target,
        )

        if key in self._edge_index:
            return self._edge_index[key]

        self.edges.append(normalized)
        self._rebuild_indexes()

        return normalized

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    def node(
        self,
        name: str,
    ) -> DependencyNode | None:
        return self._node_index.get(name)

    def get_node(
        self,
        name: str,
    ) -> DependencyNode:
        try:
            return self._node_index[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dependency node: {name}"
            ) from exc

    def has_node(self, name: str) -> bool:
        return name in self._node_index

    # ------------------------------------------------------------------
    # Direct relationships
    # ------------------------------------------------------------------

    def downstream(
        self,
        name: str,
    ) -> list[str]:
        """Return immediate downstream dependencies."""
        return list(self._children.get(name, []))

    def upstream(
        self,
        name: str,
    ) -> list[str]:
        """Return immediate upstream dependencies."""
        return list(self._parents.get(name, []))

    # Compatibility names from the original graph.
    def children(
        self,
        node_id: str,
    ) -> list[str]:
        return self.downstream(node_id)

    def parents(
        self,
        node_id: str,
    ) -> list[str]:
        return self.upstream(node_id)

    # ------------------------------------------------------------------
    # Recursive relationships
    # ------------------------------------------------------------------

    def descendants(
        self,
        name: str,
    ) -> set[str]:
        """
        Return every node downstream from ``name``.
        """
        seen: set[str] = set()

        queue = deque(
            self.downstream(name)
        )

        while queue:
            current = queue.popleft()

            if current in seen:
                continue

            seen.add(current)

            queue.extend(
                self.downstream(current)
            )

        return seen

    def ancestors(
        self,
        name: str,
    ) -> set[str]:
        """
        Return every node upstream from ``name``.
        """
        seen: set[str] = set()

        queue = deque(
            self.upstream(name)
        )

        while queue:
            current = queue.popleft()

            if current in seen:
                continue

            seen.add(current)

            queue.extend(
                self.upstream(current)
            )

        return seen

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def path_distance(
        self,
        source: str,
        target: str,
    ) -> int | None:
        """
        Return shortest directed path length.

        Returns ``None`` when no path exists.
        """
        if source == target:
            return 0

        queue = deque(
            [(source, 0)]
        )

        seen = {source}

        while queue:
            current, distance = queue.popleft()

            for child in self.downstream(current):
                if child in seen:
                    continue

                if child == target:
                    return distance + 1

                seen.add(child)

                queue.append(
                    (
                        child,
                        distance + 1,
                    )
                )

        return None

    def path(
        self,
        source: str,
        target: str,
    ) -> list[str] | None:
        """
        Return one shortest directed path between two components.
        """
        if source == target:
            return [source]

        queue = deque([source])
        previous: dict[str, str | None] = {
            source: None
        }

        while queue:
            current = queue.popleft()

            for child in self.downstream(current):
                if child in previous:
                    continue

                previous[child] = current

                if child == target:
                    result = [target]
                    cursor = target

                    while previous[cursor] is not None:
                        cursor = previous[cursor]  # type: ignore[assignment]
                        result.append(cursor)

                    result.reverse()
                    return result

                queue.append(child)

        return None

    # ------------------------------------------------------------------
    # Blast radius
    # ------------------------------------------------------------------

    def affected_components(
        self,
        components: Iterable[str],
        *,
        include_sources: bool = True,
    ) -> set[str]:
        """
        Calculate downstream blast radius.

        ``components`` are treated as failed/affected components.
        """
        affected: set[str] = set()

        for component in components:
            if include_sources:
                affected.add(component)

            affected.update(
                self.descendants(component)
            )

        return affected

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the graph using the canonical API representation.
        """
        return {
            "nodes": [
                node.to_dict()
                for node in self.nodes
            ],
            "edges": [
                edge.to_dict()
                for edge in self.edges
            ],
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Serialize using the original Phase 1-4 graph contract.
        """
        legacy_nodes: list[dict[str, Any]] = []

        for node in self.nodes:
            metadata = node.metadata

            legacy_nodes.append(
                {
                    "node_id": node.name,
                    "label": metadata.get(
                        "label",
                        node.name,
                    ),
                    "kind": node.component_type,
                    "team": metadata.get(
                        "team",
                        "",
                    ),
                }
            )

        return {
            "nodes": legacy_nodes,
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relation,
                    "weight": edge.weight,
                }
                for edge in self.edges
            ],
        }


# ---------------------------------------------------------------------------
# Deployment evidence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeploymentEvent:
    """Deployment evidence that can contribute to RCA."""

    timestamp: datetime
    component: str
    version: str
    step: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.timestamp

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return {
            "timestamp": timestamp.isoformat(),
            "component": self.component,
            "version": self.version,
            "step": self.step,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# RCA hypothesis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RootCauseHypothesis:
    """Ranked RCA hypothesis with supporting evidence."""

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
            "evidence_signals": list(
                self.evidence_signals
            ),
            "downstream_effects": list(
                self.downstream_effects
            ),
        }


# ---------------------------------------------------------------------------
# RCA engine
# ---------------------------------------------------------------------------


class GraphRCAEngine:
    """
    Topology-aware root-cause ranking.

    This preserves the original project RCA algorithm while making it
    independent from the old graph data structures.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        *,
        deployment_recency_window: int = 6,
        top_k: int = 3,
    ) -> None:
        if deployment_recency_window < 0:
            raise ValueError(
                "deployment_recency_window cannot be negative."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        self.graph = graph
        self.deployment_recency_window = (
            deployment_recency_window
        )
        self.top_k = top_k

    def analyze(
        self,
        *,
        step: int,
        anomalies: list[DetectedAnomaly],
        drift_findings: list[Any] | None = None,
        quality_issues: list[Any] | None = None,
        feature_observations: list[Any] | None = None,
        deployment_events: list[DeploymentEvent] | None = None,
    ) -> list[RootCauseHypothesis]:
        """
        Rank components that could explain the observed incident.

        Monitoring objects are accepted structurally to avoid making the
        graph layer depend on one specific monitoring implementation.
        """
        drift_findings = drift_findings or []
        quality_issues = quality_issues or []
        feature_observations = feature_observations or []
        deployment_events = deployment_events or []

        signal_bucket: dict[
            str,
            list[str],
        ] = defaultdict(list)

        score_bucket: dict[
            str,
            float,
        ] = defaultdict(float)

        direct_evidence: dict[
            str,
            list[str],
        ] = defaultdict(list)

        # --------------------------------------------------------------
        # Telemetry anomalies
        # --------------------------------------------------------------

        for anomaly in anomalies:
            component = anomaly.entity

            signal_bucket[component].append(
                f"{anomaly.metric}: {anomaly.message}"
            )

            score_bucket[component] += (
                _score_anomaly(anomaly)
            )

            direct_evidence[component].append(
                anomaly.metric
            )

        # --------------------------------------------------------------
        # Drift evidence
        # --------------------------------------------------------------

        for finding in drift_findings:
            feature_name = getattr(
                finding,
                "feature_name",
                "unknown",
            )

            psi = float(
                getattr(
                    finding,
                    "psi",
                    0.0,
                )
            )

            component = getattr(
                finding,
                "entity",
                "feature-service",
            )

            signal_bucket[component].append(
                f"drift:{feature_name} psi={psi:.3f}"
            )

            score_bucket[component] += min(
                0.28,
                psi * 0.42,
            )

            direct_evidence[component].append(
                f"drift:{feature_name}"
            )

        # --------------------------------------------------------------
        # Data-quality evidence
        # --------------------------------------------------------------

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

            rate = float(
                getattr(
                    issue,
                    "rate",
                    0.0,
                )
            )

            component = getattr(
                issue,
                "entity",
                "feature-service",
            )

            signal_bucket[component].append(
                f"dq:{feature_name} {issue_type}"
            )

            score_bucket[component] += (
                _score_quality_issue(
                    issue_type=issue_type,
                    rate=rate,
                )
            )

            direct_evidence[component].append(
                f"dq:{feature_name}"
            )

        # --------------------------------------------------------------
        # Feature observations
        # --------------------------------------------------------------

        for observation in feature_observations:
            value = getattr(
                observation,
                "value",
                None,
            )

            if value is None:
                continue

            component = getattr(
                observation,
                "entity",
                "feature-service",
            )

            feature_name = getattr(
                observation,
                "feature_name",
                "unknown",
            )

            signal_bucket[component].append(
                f"feature:{feature_name}"
            )

            score_bucket[component] += 0.01

        # --------------------------------------------------------------
        # Deployment evidence
        # --------------------------------------------------------------

        active_deployments = [
            event
            for event in deployment_events
            if step - event.step
            <= self.deployment_recency_window
        ]

        for deployment in active_deployments:
            component = deployment.component

            signal_bucket[component].append(
                f"deploy:{deployment.version}"
            )

            score_bucket[component] += 0.34

            direct_evidence[component].append(
                f"deploy:{deployment.version}"
            )

        # --------------------------------------------------------------
        # Graph topology
        # --------------------------------------------------------------

        for node in self.graph.nodes:
            node_id = node.name

            descendants = self.graph.descendants(
                node_id
            )

            impacted_descendants = [
                descendant
                for descendant in descendants
                if descendant in score_bucket
            ]

            if impacted_descendants:
                score_bucket[node_id] += min(
                    0.22,
                    0.05
                    * len(impacted_descendants),
                )

                for descendant in impacted_descendants:
                    distance = (
                        self.graph.path_distance(
                            node_id,
                            descendant,
                        )
                        or 1
                    )

                    score_bucket[node_id] += max(
                        0.0,
                        0.08 / distance,
                    )

            if (
                node_id in score_bucket
                and self.graph.downstream(node_id)
            ):
                score_bucket[node_id] += 0.05

            if node_id in score_bucket:
                ancestors = self.graph.ancestors(
                    node_id
                )

                score_bucket[node_id] += min(
                    0.08,
                    0.02
                    * len(ancestors),
                )

        # --------------------------------------------------------------
        # Build hypotheses
        # --------------------------------------------------------------

        hypotheses: list[
            RootCauseHypothesis
        ] = []

        ranked = sorted(
            score_bucket.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for node_id, score in ranked:
            node = self.graph.node(node_id)

            if node is None:
                continue

            label = str(
                node.metadata.get(
                    "label",
                    node.name,
                )
            )

            hypotheses.append(
                RootCauseHypothesis(
                    component=node.name,
                    label=label,
                    score=round(
                        min(score, 1.0),
                        3,
                    ),
                    confidence=round(
                        _confidence_from_score(score),
                        2,
                    ),
                    reasons=_reasons_for_component(
                        node=node,
                        score=score,
                        signals=signal_bucket.get(
                            node_id,
                            [],
                        ),
                        deployment_events=active_deployments,
                        graph=self.graph,
                    ),
                    evidence_signals=sorted(
                        set(
                            direct_evidence.get(
                                node_id,
                                [],
                            )
                        )
                    ),
                    downstream_effects=sorted(
                        self.graph.descendants(
                            node_id
                        )
                    ),
                )
            )

        if not hypotheses:
            return [
                RootCauseHypothesis(
                    component="unknown",
                    label="Insufficient evidence",
                    score=0.0,
                    confidence=0.0,
                    reasons=[
                        "No telemetry, drift, "
                        "quality, or deployment "
                        "evidence is available."
                    ],
                    evidence_signals=[],
                    downstream_effects=[],
                )
            ]

        return hypotheses[: self.top_k]


# ---------------------------------------------------------------------------
# Canonical RootCauseAnalyzer
# ---------------------------------------------------------------------------


class RootCauseAnalyzer:
    """
    Canonical RCA service.

    ``GraphRCAEngine`` remains available for compatibility, while this class
    provides the target application-level API.
    """

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        historical_incidents: Iterable[Any] = (),
        *,
        top_k: int = 3,
    ) -> None:
        self.dependency_graph = dependency_graph
        self.historical_incidents = list(
            historical_incidents
        )

        self.engine = GraphRCAEngine(
            dependency_graph,
            top_k=top_k,
        )

    def analyze(
        self,
        incident: Any | None = None,
        *,
        anomalies: list[DetectedAnomaly] | None = None,
        step: int = 0,
        drift_findings: list[Any] | None = None,
        quality_issues: list[Any] | None = None,
        feature_observations: list[Any] | None = None,
        deployment_events: list[DeploymentEvent] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze an incident and return structured RCA results.
        """
        if anomalies is None:
            anomalies = list(
                getattr(
                    incident,
                    "anomalies",
                    [],
                )
                if incident is not None
                else []
            )

        hypotheses = self.engine.analyze(
            step=step,
            anomalies=anomalies,
            drift_findings=drift_findings,
            quality_issues=quality_issues,
            feature_observations=feature_observations,
            deployment_events=deployment_events,
        )

        top = hypotheses[0]

        return {
            "root_cause": top.component,
            "label": top.label,
            "confidence": top.confidence,
            "score": top.score,
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in hypotheses
            ],
            "downstream_effects": sorted(
                self.dependency_graph.descendants(
                    top.component
                )
            )
            if top.component != "unknown"
            else [],
        }

    def rank_root_causes(
        self,
        hypotheses: Iterable[
            RootCauseHypothesis
        ],
    ) -> list[RootCauseHypothesis]:
        """Return hypotheses sorted by score."""
        return sorted(
            hypotheses,
            key=lambda item: (
                item.score,
                item.confidence,
            ),
            reverse=True,
        )

    def find_historical_matches(
        self,
        incident: Any,
        *,
        limit: int = 5,
    ) -> list[Any]:
        """
        Find simple historical matches.

        Matching intentionally uses deterministic component/metric overlap.
        A vector database is unnecessary for this local platform.
        """
        if limit < 1:
            return []

        incident_component = getattr(
            incident,
            "component",
            None,
        )

        incident_metrics = {
            anomaly.metric
            for anomaly in getattr(
                incident,
                "anomalies",
                [],
            )
        }

        scored: list[
            tuple[float, Any]
        ] = []

        for historical in self.historical_incidents:
            historical_component = getattr(
                historical,
                "component",
                None,
            )

            historical_anomalies = getattr(
                historical,
                "anomalies",
                [],
            )

            historical_metrics = {
                getattr(
                    anomaly,
                    "metric",
                    "",
                )
                for anomaly in historical_anomalies
            }

            component_match = (
                1.0
                if (
                    incident_component
                    and historical_component
                    == incident_component
                )
                else 0.0
            )

            if incident_metrics or historical_metrics:
                union = (
                    incident_metrics
                    | historical_metrics
                )

                intersection = (
                    incident_metrics
                    & historical_metrics
                )

                metric_match = (
                    len(intersection)
                    / len(union)
                    if union
                    else 0.0
                )
            else:
                metric_match = 0.0

            score = (
                0.6 * component_match
                + 0.4 * metric_match
            )

            if score > 0:
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
            historical
            for _, historical in scored[:limit]
        ]

    def explain_chain(
        self,
        root_cause: str,
        affected_component: str | None = None,
    ) -> list[str]:
        """
        Explain the dependency propagation chain.
        """
        if affected_component is None:
            descendants = sorted(
                self.dependency_graph.descendants(
                    root_cause
                )
            )

            return [
                f"{root_cause} -> {component}"
                for component in descendants
            ]

        path = self.dependency_graph.path(
            root_cause,
            affected_component,
        )

        if path is None:
            return [
                (
                    f"No dependency path from "
                    f"{root_cause} to "
                    f"{affected_component}."
                )
            ]

        return [
            " -> ".join(path)
        ]


# ---------------------------------------------------------------------------
# Default dependency graph
# ---------------------------------------------------------------------------


def build_default_dependency_graph() -> DependencyGraph:
    graph = DependencyGraph()

    for name, component_type in (
        ("api-gateway", "service"),
        ("feature-service", "service"),
        ("model-serving", "service"),
        ("prediction-api", "service"),
        ("data-pipeline", "pipeline"),
    ):
        graph.add_node(
            DependencyNode(
                name=name,
                component_type=component_type,
            )
        )

    graph.add_edge(
        DependencyEdge(
            source="api-gateway",
            target="feature-service",
            relation="depends_on",
            weight=1.0,
        )
    )

    graph.add_edge(
        DependencyEdge(
            source="feature-service",
            target="model-serving",
            relation="depends_on",
            weight=1.0,
        )
    )

    graph.add_edge(
        DependencyEdge(
            source="model-serving",
            target="prediction-api",
            relation="depends_on",
            weight=1.0,
        )
    )

    graph.add_edge(
        DependencyEdge(
            source="data-pipeline",
            target="feature-service",
            relation="feeds",
            weight=1.0,
        )
    )

    # Compatibility with the existing Phase 3 test suite.
    graph.add_edge(
        DependencyEdge(
            source="feature-service",
            target="api-gateway",
            relation="serves",
            weight=0.8,
        )
    )

    return graph


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_anomaly(
    anomaly: DetectedAnomaly,
) -> float:
    severity_weight = {
        "critical": 0.36,
        "high": 0.28,
        "medium": 0.20,
        "low": 0.12,
    }.get(
        anomaly.severity,
        0.18,
    )

    metric_weight = {
        "prediction_latency_ms": 0.20,
        "feature_missing_rate": 0.22,
        "error_rate": 0.18,
        "prediction_ctr": 0.18,
        "traffic_rps": 0.10,
    }.get(
        anomaly.metric,
        0.10,
    )

    return (
        severity_weight
        + metric_weight
    )


def _score_quality_issue(
    *,
    issue_type: str,
    rate: float,
) -> float:
    issue_weight = {
        "missing_rate": 0.24,
        "out_of_range": 0.20,
        "duplicate_request": 0.10,
    }.get(
        issue_type,
        0.08,
    )

    return (
        issue_weight
        + min(
            0.18,
            rate * 0.34,
        )
    )


def _confidence_from_score(
    score: float,
) -> float:
    return min(
        0.99,
        0.34
        + 0.72
        * (
            1.0
            - exp(-score)
        ),
    )


def _reasons_for_component(
    *,
    node: DependencyNode,
    score: float,
    signals: list[str],
    deployment_events: list[DeploymentEvent],
    graph: DependencyGraph,
) -> list[str]:
    reasons: list[str] = []

    deployment = next(
        (
            event
            for event in deployment_events
            if event.component == node.name
        ),
        None,
    )

    if deployment is not None:
        reasons.append(
            f"Deployment "
            f"{deployment.version} "
            f"landed on "
            f"{deployment.component} "
            f"at step "
            f"{deployment.step}."
        )

    if signals:
        label = str(
            node.metadata.get(
                "label",
                node.name,
            )
        )

        reasons.append(
            f"Direct evidence observed on "
            f"{label}: "
            f"{', '.join(signals[:3])}."
        )

    descendants = sorted(
        graph.descendants(
            node.name
        )
    )

    if descendants:
        label = str(
            node.metadata.get(
                "label",
                node.name,
            )
        )

        reasons.append(
            f"{label} sits upstream of "
            f"{', '.join(descendants)} "
            f"in the dependency graph."
        )

    if score >= 0.8:
        reasons.append(
            "Strong combined evidence from "
            "telemetry, drift, quality, "
            "deployment, and graph topology."
        )
    elif score >= 0.5:
        reasons.append(
            "Multiple signals support this "
            "hypothesis, though some uncertainty "
            "remains."
        )
    else:
        reasons.append(
            "Weak supporting evidence; included "
            "as a lower-ranked candidate."
        )

    return reasons


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "DependencyNode",
    "DependencyEdge",
    "DependencyGraph",
    "GraphNode",
    "GraphEdge",
    "DeploymentEvent",
    "RootCauseHypothesis",
    "GraphRCAEngine",
    "RootCauseAnalyzer",
    "build_default_dependency_graph",
]
