from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

VALID_SEVERITIES = frozenset(
    {
        SEVERITY_LOW,
        SEVERITY_MEDIUM,
        SEVERITY_HIGH,
        SEVERITY_CRITICAL,
    }
)

STATUS_OPEN = "OPEN"
STATUS_INVESTIGATING = "INVESTIGATING"
STATUS_RESOLVED = "RESOLVED"

VALID_STATUSES = frozenset(
    {
        STATUS_OPEN,
        STATUS_INVESTIGATING,
        STATUS_RESOLVED,
    }
)

INVESTIGATION_LOCAL_FALLBACK = "local-fallback"
INVESTIGATION_OLLAMA = "ollama"
INVESTIGATION_OPENAI = "openai"

VALID_INVESTIGATION_MODES = frozenset(
    {
        INVESTIGATION_LOCAL_FALLBACK,
        INVESTIGATION_OLLAMA,
        INVESTIGATION_OPENAI,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _serialize_datetime(value: datetime) -> str:
    """Serialize datetime consistently for API/dashboard responses."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.isoformat()


def _copy_string_dict(value: dict[str, str]) -> dict[str, str]:
    """Return a defensive copy of a string dictionary."""
    return dict(value)


def _copy_string_list(value: list[str]) -> list[str]:
    """Return a defensive copy of a string list."""
    return list(value)


# ---------------------------------------------------------------------------
# Metric telemetry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MetricSample:
    """
    Single telemetry sample.

    A sample represents one observation of one metric for one entity.
    """

    timestamp: datetime
    metric: str
    entity: str
    value: float
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the sample into a JSON-safe dictionary."""
        return {
            "timestamp": _serialize_datetime(self.timestamp),
            "metric": self.metric,
            "entity": self.entity,
            "value": self.value,
            "unit": self.unit,
            "tags": _copy_string_dict(self.tags),
        }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DetectedAnomaly:
    """
    Detector-produced anomaly.

    ``baseline`` is the expected operating value at the time of detection.
    ``z_score`` intentionally keeps its sign:

        positive -> observed value is above baseline
        negative -> observed value is below baseline

    Severity is derived from ``abs(z_score)`` by the detector and/or
    incident correlation layer.
    """

    metric: str
    entity: str
    value: float
    baseline: float
    z_score: float
    severity: str
    timestamp: datetime
    detector: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid anomaly severity: {self.severity!r}. "
                f"Expected one of: {sorted(VALID_SEVERITIES)}"
            )

    # ------------------------------------------------------------------
    # Backward-compatible aliases for the existing Phase 1-4 code.
    # ------------------------------------------------------------------

    @property
    def expected_mean(self) -> float:
        """
        Backward-compatible alias for the old ``expected_mean`` field.

        The canonical name is ``baseline``.
        """
        return self.baseline

    @property
    def expected_std(self) -> float:
        """
        Backward-compatible compatibility value.

        The new domain model does not require expected standard deviation.
        A detector that needs this value should keep it internally.

        This property exists temporarily for older dashboard/investigation
        code that expects the field to exist.
        """
        if self.z_score == 0:
            return 0.0

        return abs(self.value - self.baseline) / abs(self.z_score)

    @property
    def detector_name(self) -> str:
        """Backward-compatible alias for ``detector``."""
        return self.detector

    @property
    def explanation(self) -> str:
        """Backward-compatible alias for ``message``."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to the canonical API representation.

        Legacy aliases are deliberately not emitted. New API consumers
        should use the canonical field names.
        """
        return {
            "metric": self.metric,
            "entity": self.entity,
            "value": self.value,
            "baseline": self.baseline,
            "z_score": self.z_score,
            "severity": self.severity,
            "timestamp": _serialize_datetime(self.timestamp),
            "detector": self.detector,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Incident:
    """
    Aggregated ML/system incident.

    An Incident represents a lifecycle object rather than a single event.

    ``started_at`` identifies when the correlated incident began.
    ``updated_at`` changes as additional evidence arrives or the incident
    is resolved.
    """

    incident_id: str
    title: str
    severity: str
    status: str
    started_at: datetime
    updated_at: datetime
    component: str
    root_cause: str | None
    confidence: float
    anomalies: list[DetectedAnomaly] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid incident severity: {self.severity!r}. "
                f"Expected one of: {sorted(VALID_SEVERITIES)}"
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid incident status: {self.status!r}. "
                f"Expected one of: {sorted(VALID_STATUSES)}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Incident confidence must be between 0.0 and 1.0, "
                f"got {self.confidence!r}"
            )

    # ------------------------------------------------------------------
    # Backward-compatible aliases for the existing Phase 1-4 code.
    # ------------------------------------------------------------------

    @property
    def detected_at(self) -> datetime:
        """
        Backward-compatible alias.

        The canonical lifecycle field is ``started_at``.
        """
        return self.started_at

    @property
    def summary(self) -> str:
        """
        Backward-compatible summary.

        The canonical Incident model intentionally does not duplicate an
        investigation summary. InvestigationReport owns that information.

        Returning the title keeps older consumers functional without
        inventing RCA information.
        """
        return self.title

    @property
    def observed_anomalies(self) -> list[DetectedAnomaly]:
        """Backward-compatible alias for ``anomalies``."""
        return self.anomalies

    @property
    def recommended_actions(self) -> list[str]:
        """Backward-compatible alias for ``recommendations``."""
        return self.recommendations

    def add_anomaly(self, anomaly: DetectedAnomaly) -> None:
        """
        Add evidence to the incident and update its timestamp.

        Duplicate anomaly objects are not added twice.
        """
        if anomaly not in self.anomalies:
            self.anomalies.append(anomaly)

        self.updated_at = utcnow()

    def mark_updated(self, timestamp: datetime | None = None) -> None:
        """Update the incident timestamp."""
        self.updated_at = timestamp or utcnow()

    def resolve(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Mark the incident as resolved."""
        self.status = STATUS_RESOLVED
        self.updated_at = timestamp or utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert the incident into a JSON-safe canonical dictionary."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity,
            "status": self.status,
            "started_at": _serialize_datetime(self.started_at),
            "updated_at": _serialize_datetime(self.updated_at),
            "component": self.component,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "anomalies": [
                anomaly.to_dict()
                for anomaly in self.anomalies
            ],
            "affected_components": _copy_string_list(
                self.affected_components
            ),
            "recommendations": _copy_string_list(
                self.recommendations
            ),
        }


# ---------------------------------------------------------------------------
# Investigation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class InvestigationReport:
    """
    Result of deterministic RCA + optional LLM investigation.

    ``mode`` identifies how the report was produced:

        local-fallback
        ollama
        openai
    """

    incident_id: str
    mode: str
    verdict: str
    confidence: float
    summary: str
    root_cause_analysis: str
    evidence: list[str] = field(default_factory=list)
    downstream_effects: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    historical_matches: list[str] = field(default_factory=list)
    usage: dict[str, object] = field(default_factory=dict)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in VALID_INVESTIGATION_MODES:
            raise ValueError(
                f"Invalid investigation mode: {self.mode!r}. "
                f"Expected one of: "
                f"{sorted(VALID_INVESTIGATION_MODES)}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Investigation confidence must be between 0.0 and 1.0, "
                f"got {self.confidence!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report into a JSON-safe dictionary.

        ``usage`` is copied defensively and must never contain secrets such
        as API keys or authorization headers.
        """
        return {
            "incident_id": self.incident_id,
            "mode": self.mode,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "root_cause_analysis": self.root_cause_analysis,
            "evidence": _copy_string_list(self.evidence),
            "downstream_effects": _copy_string_list(
                self.downstream_effects
            ),
            "recommendations": _copy_string_list(
                self.recommendations
            ),
            "historical_matches": _copy_string_list(
                self.historical_matches
            ),
            "usage": dict(self.usage),
        }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "MetricSample",
    "DetectedAnomaly",
    "Incident",
    "InvestigationReport",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_HIGH",
    "SEVERITY_CRITICAL",
    "VALID_SEVERITIES",
    "STATUS_OPEN",
    "STATUS_INVESTIGATING",
    "STATUS_RESOLVED",
    "VALID_STATUSES",
    "INVESTIGATION_LOCAL_FALLBACK",
    "INVESTIGATION_OLLAMA",
    "INVESTIGATION_OPENAI",
    "VALID_INVESTIGATION_MODES",
    "utcnow",
]
