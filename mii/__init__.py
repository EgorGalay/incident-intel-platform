"""Import shim so the `src/` layout works from a plain repository checkout."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _ROOT.parent / "src" / "mii"

if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

from .detectors import DetectorSuite, EWMADeviationDetector, RollingZScoreDetector
from .graph import (
    DeploymentEvent,
    DependencyGraph,
    GraphEdge,
    GraphNode,
    GraphRCAEngine,
    RootCauseHypothesis,
    build_default_dependency_graph,
)
from .incidents import IncidentEngine
from .monitoring import DataQualityIssue, DataQualityMonitor, DriftFinding, DriftMonitor, FeatureObservation
from .models import DetectedAnomaly, Incident, MetricSample
from .synthetic import FeatureProfile, FaultProfile, SyntheticWorkload
from .state import PhaseOneRuntime

__all__ = [
    "DataQualityIssue",
    "DataQualityMonitor",
    "DeploymentEvent",
    "DependencyGraph",
    "DetectedAnomaly",
    "DetectorSuite",
    "DriftFinding",
    "DriftMonitor",
    "EWMADeviationDetector",
    "FeatureObservation",
    "FeatureProfile",
    "FaultProfile",
    "GraphEdge",
    "GraphNode",
    "GraphRCAEngine",
    "Incident",
    "IncidentEngine",
    "MetricSample",
    "PhaseOneRuntime",
    "RootCauseHypothesis",
    "RollingZScoreDetector",
    "SyntheticWorkload",
    "build_default_dependency_graph",
]
