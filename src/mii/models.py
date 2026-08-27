from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class MetricSample:
    """Single telemetry sample for one metric."""

    timestamp: datetime
    metric: str
    entity: str
    value: float
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric,
            "entity": self.entity,
            "value": self.value,
            "unit": self.unit,
            "tags": dict(self.tags),
        }


@dataclass(slots=True)
class DetectedAnomaly:
    """A detector-produced anomaly with an explanation."""

    timestamp: datetime
    metric: str
    entity: str
    value: float
    expected_mean: float
    expected_std: float
    z_score: float
    detector_name: str
    explanation: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric,
            "entity": self.entity,
            "value": self.value,
            "expected_mean": self.expected_mean,
            "expected_std": self.expected_std,
            "z_score": self.z_score,
            "detector_name": self.detector_name,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass(slots=True)
class Incident:
    """Aggregated incident created from correlated anomalies."""

    incident_id: str
    title: str
    detected_at: datetime
    severity: str
    confidence: float
    summary: str
    observed_anomalies: list[DetectedAnomaly]
    recommended_actions: list[str]
    status: str = "OPEN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "detected_at": self.detected_at.isoformat(),
            "severity": self.severity,
            "confidence": self.confidence,
            "summary": self.summary,
            "observed_anomalies": [anomaly.to_dict() for anomaly in self.observed_anomalies],
            "recommended_actions": list(self.recommended_actions),
            "status": self.status,
        }
