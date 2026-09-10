from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import count
from typing import Iterable

from .models import (
    DetectedAnomaly,
    Incident,
    STATUS_INVESTIGATING,
    STATUS_OPEN,
    STATUS_RESOLVED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    utcnow,
)


# ---------------------------------------------------------------------------
# Internal correlation state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _IncidentCandidate:
    """
    Temporary correlation window.

    A candidate is not necessarily an Incident yet. It becomes an Incident
    when enough independent evidence is present.
    """

    anomalies: list[DetectedAnomaly] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    def add(self, anomaly: DetectedAnomaly) -> None:
        """Add an anomaly to the candidate."""
        if anomaly not in self.anomalies:
            self.anomalies.append(anomaly)

        if self.first_seen is None:
            self.first_seen = anomaly.timestamp

        if self.last_seen is None or anomaly.timestamp > self.last_seen:
            self.last_seen = anomaly.timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_timestamp(value: datetime) -> datetime:
    """Return a timezone-aware timestamp."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value


def _severity_rank(severity: str) -> int:
    """Return numeric severity ordering."""
    return {
        SEVERITY_LOW: 1,
        SEVERITY_MEDIUM: 2,
        SEVERITY_HIGH: 3,
        SEVERITY_CRITICAL: 4,
    }.get(severity, 0)


def _max_severity(
    anomalies: Iterable[DetectedAnomaly],
) -> str:
    """Return the highest severity among anomalies."""
    severities = [anomaly.severity for anomaly in anomalies]

    if not severities:
        return SEVERITY_LOW

    return max(
        severities,
        key=_severity_rank,
    )


def _component_from_anomaly(
    anomaly: DetectedAnomaly,
) -> str:
    """
    Infer a component from an anomaly.

    The simulator normally supplies the component as entity. This helper
    keeps the incident engine independent from the simulator implementation.
    """
    if anomaly.entity:
        return anomaly.entity

    return "unknown"


def _incident_title(
    *,
    component: str,
    anomalies: list[DetectedAnomaly],
) -> str:
    """Build a deterministic human-readable incident title."""
    component_title = component.replace("-", " ")

    if component_title:
        component_title = (
            component_title[0].upper()
            + component_title[1:]
        )

    metrics = sorted(
        {
            anomaly.metric
            for anomaly in anomalies
            if anomaly.metric
        }
    )

    if not metrics:
        return f"{component_title} degradation detected"

    if len(metrics) == 1:
        return f"{component_title} anomaly: {metrics[0]}"

    return (
        f"{component_title} degradation detected "
        f"({len(metrics)} correlated metrics)"
    )


def _incident_recommendations(
    anomalies: list[DetectedAnomaly],
) -> list[str]:
    """
    Generate deterministic first-response recommendations.

    These are deliberately conservative. Detailed remediation belongs to
    RCA/investigation.
    """
    recommendations: list[str] = []

    metrics = {
        anomaly.metric
        for anomaly in anomalies
    }

    if "prediction_latency_ms" in metrics:
        recommendations.append(
            "Inspect model-serving latency and recent deployment changes."
        )

    if "error_rate" in metrics:
        recommendations.append(
            "Inspect application errors and downstream service health."
        )

    if "feature_missing_rate" in metrics:
        recommendations.append(
            "Inspect the feature/data pipeline for missing or delayed data."
        )

    if "traffic_rps" in metrics:
        recommendations.append(
            "Check upstream traffic sources and API gateway health."
        )

    if "prediction_ctr" in metrics:
        recommendations.append(
            "Check model output quality and recent feature distribution changes."
        )

    if not recommendations:
        recommendations.append(
            "Inspect correlated anomalies and dependency health."
        )

    return recommendations


def _affected_components(
    anomalies: Iterable[DetectedAnomaly],
) -> list[str]:
    """Return unique affected components in deterministic order."""
    components = {
        _component_from_anomaly(anomaly)
        for anomaly in anomalies
        if _component_from_anomaly(anomaly)
    }

    return sorted(components)


def _incident_confidence(
    anomalies: list[DetectedAnomaly],
) -> float:
    """
    Estimate deterministic confidence from correlated evidence.

    This is not a probability. It is a normalized evidence-strength score.
    """
    if not anomalies:
        return 0.0

    distinct_metrics = len(
        {
            anomaly.metric
            for anomaly in anomalies
        }
    )

    distinct_components = len(
        {
            anomaly.entity
            for anomaly in anomalies
        }
    )

    severity_bonus = (
        max(
            _severity_rank(anomaly.severity)
            for anomaly in anomalies
        )
        / 4.0
    )

    metric_score = min(distinct_metrics / 3.0, 1.0)
    component_score = min(distinct_components / 3.0, 1.0)

    score = (
        0.45 * metric_score
        + 0.25 * component_score
        + 0.30 * severity_bonus
    )

    return min(max(score, 0.0), 1.0)


def _dedup_key(anomaly: DetectedAnomaly) -> tuple[str, str, str]:
    """
    Return a stable identity for an anomaly.

    Value/timestamp are intentionally excluded so repeated observations of
    the same signal can be correlated into the same incident.
    """
    return (
        anomaly.entity,
        anomaly.metric,
        anomaly.detector,
    )


# ---------------------------------------------------------------------------
# Incident engine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IncidentEngine:
    """
    Correlate anomalies into incidents and manage their lifecycle.

    The engine is intentionally in-memory because this project is a local
    ML incident intelligence platform. Persistence can be introduced later
    without changing the domain API.

    Correlation rules
    -----------------
    1. Anomalies are grouped by temporal proximity.
    2. A candidate must contain enough evidence before becoming an incident.
    3. Existing incidents absorb new related anomalies.
    4. Duplicate signals inside an active incident are ignored.
    5. Incidents can automatically resolve after a quiet period.
    """

    minimum_metrics: int = 2
    minimum_anomalies: int = 3
    correlation_window_seconds: int = 180
    incident_ttl_seconds: int = 600
    resolution_quiet_seconds: int = 120
    max_incidents: int = 100

    # Compatibility with the old implementation.
    correlation_window: int | None = None

    _counter: count = field(
        default_factory=lambda: count(1),
        init=False,
        repr=False,
    )

    _candidates: deque[_IncidentCandidate] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )

    _incidents: dict[str, Incident] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _dedup: dict[
        tuple[str, str, str],
        str,
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.correlation_window is not None:
            self.correlation_window_seconds = (
                self.correlation_window * 60
            )

        if self.minimum_metrics < 1:
            raise ValueError(
                "minimum_metrics must be at least 1."
            )

        if self.minimum_anomalies < 1:
            raise ValueError(
                "minimum_anomalies must be at least 1."
            )

        if self.correlation_window_seconds <= 0:
            raise ValueError(
                "correlation_window_seconds must be positive."
            )

        if self.incident_ttl_seconds <= 0:
            raise ValueError(
                "incident_ttl_seconds must be positive."
            )

        if self.resolution_quiet_seconds <= 0:
            raise ValueError(
                "resolution_quiet_seconds must be positive."
            )

        if self.max_incidents < 1:
            raise ValueError(
                "max_incidents must be at least 1."
            )

    # ------------------------------------------------------------------
    # Public views
    # ------------------------------------------------------------------

    @property
    def incidents(self) -> list[Incident]:
        """Return all retained incidents."""
        return list(self._incidents.values())

    @property
    def active_incidents(self) -> list[Incident]:
        """Return incidents that are not resolved."""
        return [
            incident
            for incident in self._incidents.values()
            if incident.status != STATUS_RESOLVED
        ]

    @property
    def active_incident(self) -> Incident | None:
        """
        Backward-compatible singular active incident property.

        New code should use ``active_incidents``.
        """
        active = self.active_incidents

        if not active:
            return None

        return max(
            active,
            key=lambda incident: incident.updated_at,
        )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _candidate_for(
        self,
        anomaly: DetectedAnomaly,
    ) -> _IncidentCandidate:
        """
        Find an existing temporal candidate or create a new one.
        """
        timestamp = _normalize_timestamp(anomaly.timestamp)
        window = timedelta(
            seconds=self.correlation_window_seconds
        )

        for candidate in reversed(self._candidates):
            if candidate.last_seen is None:
                continue

            if timestamp - candidate.last_seen <= window:
                return candidate

        candidate = _IncidentCandidate()
        self._candidates.append(candidate)

        return candidate

    def _cleanup_candidates(
        self,
        now: datetime,
    ) -> None:
        """Remove expired correlation candidates."""
        window = timedelta(
            seconds=self.correlation_window_seconds
        )

        retained: deque[_IncidentCandidate] = deque()

        for candidate in self._candidates:
            if candidate.last_seen is None:
                continue

            if now - candidate.last_seen <= window:
                retained.append(candidate)

        self._candidates = retained

    def _find_related_incident(
        self,
        anomaly: DetectedAnomaly,
    ) -> Incident | None:
        """
        Find an active incident related to this anomaly.

        Relation is based on component plus temporal proximity.
        """
        timestamp = _normalize_timestamp(anomaly.timestamp)

        window = timedelta(
            seconds=self.correlation_window_seconds
        )

        best: Incident | None = None

        for incident in self.active_incidents:
            incident_components = {
                incident.component,
                *incident.affected_components,
            }

            if anomaly.entity not in incident_components:
                continue

            if abs(timestamp - incident.updated_at) > window:
                continue

            if best is None or incident.updated_at > best.updated_at:
                best = incident

        return best

    # ------------------------------------------------------------------
    # Incident creation/update
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        """Generate a deterministic incident identifier."""
        return f"INC-{next(self._counter):04d}"

    def _create_incident(
        self,
        anomalies: list[DetectedAnomaly],
    ) -> Incident:
        """Create a new incident from correlated anomalies."""
        anomalies = list(anomalies)

        component_counts: dict[str, int] = {}

        for anomaly in anomalies:
            component = _component_from_anomaly(anomaly)

            component_counts[component] = (
                component_counts.get(component, 0) + 1
            )

        component = max(
            component_counts,
            key=component_counts.get,
        )

        started_at = min(
            _normalize_timestamp(anomaly.timestamp)
            for anomaly in anomalies
        )

        updated_at = max(
            _normalize_timestamp(anomaly.timestamp)
            for anomaly in anomalies
        )

        incident = Incident(
            incident_id=self._next_id(),
            title=_incident_title(
                component=component,
                anomalies=anomalies,
            ),
            severity=_max_severity(anomalies),
            status=STATUS_OPEN,
            started_at=started_at,
            updated_at=updated_at,
            component=component,
            root_cause=None,
            confidence=_incident_confidence(anomalies),
            anomalies=list(anomalies),
            affected_components=_affected_components(anomalies),
            recommendations=_incident_recommendations(
                anomalies
            ),
        )

        self._incidents[incident.incident_id] = incident

        for anomaly in anomalies:
            self._dedup[_dedup_key(anomaly)] = (
                incident.incident_id
            )

        self._trim_incidents()

        return incident

    def _update_incident(
        self,
        incident: Incident,
        anomaly: DetectedAnomaly,
    ) -> Incident:
        """Merge a new anomaly into an existing incident."""
        key = _dedup_key(anomaly)

        if self._dedup.get(key) == incident.incident_id:
            # Same signal family already belongs to this incident.
            # Still refresh lifecycle time because the system remains active.
            incident.mark_updated(
                _normalize_timestamp(anomaly.timestamp)
            )
            return incident

        incident.add_anomaly(anomaly)

        if anomaly.entity not in incident.affected_components:
            incident.affected_components.append(
                anomaly.entity
            )
            incident.affected_components.sort()

        candidate_severity = _max_severity(
            incident.anomalies
        )

        if (
            _severity_rank(candidate_severity)
            > _severity_rank(incident.severity)
        ):
            incident.severity = candidate_severity

        incident.confidence = _incident_confidence(
            incident.anomalies
        )

        incident.recommendations = _incident_recommendations(
            incident.anomalies
        )

        self._dedup[key] = incident.incident_id

        return incident

    def _trim_incidents(self) -> None:
        """Bound retained incident history."""
        if len(self._incidents) <= self.max_incidents:
            return

        ordered = sorted(
            self._incidents.values(),
            key=lambda incident: incident.updated_at,
        )

        while len(ordered) > self.max_incidents:
            oldest = ordered.pop(0)
            self._incidents.pop(
                oldest.incident_id,
                None,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def investigate(
        self,
        incident_id: str,
    ) -> Incident:
        """Move an incident into INVESTIGATING state."""
        incident = self.get_incident(incident_id)

        if incident.status == STATUS_RESOLVED:
            raise ValueError(
                f"Cannot investigate resolved incident "
                f"{incident_id!r}."
            )

        incident.status = STATUS_INVESTIGATING
        incident.mark_updated()

        return incident

    def resolve(
        self,
        incident_id: str,
        *,
        timestamp: datetime | None = None,
    ) -> Incident:
        """Explicitly resolve an incident."""
        incident = self.get_incident(incident_id)

        incident.resolve(
            timestamp=_normalize_timestamp(timestamp)
            if timestamp is not None
            else None
        )

        return incident

    def _auto_resolve(
        self,
        now: datetime,
    ) -> list[Incident]:
        """
        Automatically resolve incidents that have been quiet long enough.
        """
        quiet_period = timedelta(
            seconds=self.resolution_quiet_seconds
        )

        resolved: list[Incident] = []

        for incident in self.active_incidents:
            if now - incident.updated_at < quiet_period:
                continue

            incident.resolve(timestamp=now)
            resolved.append(incident)

        return resolved

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def observe(
        self,
        step: int | None = None,
        anomalies: Iterable[DetectedAnomaly] = (),
        timestamp: datetime | None = None,
    ) -> Incident | None:
        """
        Observe anomalies and return the incident affected by this batch.

        ``step`` is retained for compatibility with the previous runtime.
        The canonical correlation logic uses anomaly timestamps.
        """
        del step

        anomaly_list = list(anomalies)

        if timestamp is not None:
            normalized_timestamp = _normalize_timestamp(timestamp)
        elif anomaly_list:
            normalized_timestamp = max(
                _normalize_timestamp(
                    anomaly.timestamp
                )
                for anomaly in anomaly_list
            )
        else:
            normalized_timestamp = utcnow()

        self._cleanup_candidates(normalized_timestamp)

        # No anomalies means only lifecycle maintenance.
        if not anomaly_list:
            self._auto_resolve(normalized_timestamp)
            return None

        affected_incident: Incident | None = None

        for anomaly in anomaly_list:
            existing = self._find_related_incident(
                anomaly
            )

            if existing is not None:
                affected_incident = self._update_incident(
                    existing,
                    anomaly,
                )
                continue

            key = _dedup_key(anomaly)

            # A signal already assigned to a resolved incident should not
            # immediately recreate the same incident unless the old incident
            # is outside the TTL.
            previous_incident_id = self._dedup.get(key)

            if previous_incident_id is not None:
                previous = self._incidents.get(
                    previous_incident_id
                )

                if previous is not None:
                    if (
                        normalized_timestamp
                        - previous.updated_at
                        <= timedelta(
                            seconds=self.incident_ttl_seconds
                        )
                    ):
                        continue

            candidate = self._candidate_for(anomaly)
            candidate.add(anomaly)

            metrics = {
                item.metric
                for item in candidate.anomalies
            }

            if (
                len(candidate.anomalies)
                < self.minimum_anomalies
                and len(metrics)
                < self.minimum_metrics
            ):
                continue

            # If minimum_anomalies is satisfied, or there are enough distinct
            # metrics, promote the candidate to an incident.
            if (
                len(candidate.anomalies)
                >= self.minimum_anomalies
                or len(metrics)
                >= self.minimum_metrics
            ):
                affected_incident = self._create_incident(
                    candidate.anomalies
                )

                # Candidate has now been promoted. Remove it so future
                # observations are correlated against the real incident.
                try:
                    self._candidates.remove(candidate)
                except ValueError:
                    pass

        self._auto_resolve(normalized_timestamp)

        return affected_incident

    def process(
        self,
        anomalies: Iterable[DetectedAnomaly],
        *,
        timestamp: datetime | None = None,
    ) -> Incident | None:
        """
        Canonical alias for ``observe``.

        Useful for code that does not have a simulation step number.
        """
        return self.observe(
            step=None,
            anomalies=anomalies,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_incident(
        self,
        incident_id: str,
    ) -> Incident:
        """Return an incident or raise KeyError."""
        try:
            return self._incidents[incident_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown incident: {incident_id}"
            ) from exc

    def get_active_incidents(self) -> list[Incident]:
        """Return all active incidents."""
        return self.active_incidents

    def get_resolved_incidents(self) -> list[Incident]:
        """Return all resolved incidents."""
        return [
            incident
            for incident in self._incidents.values()
            if incident.status == STATUS_RESOLVED
        ]

    def get_recent_incidents(
        self,
        *,
        limit: int = 20,
    ) -> list[Incident]:
        """Return incidents ordered from newest to oldest."""
        if limit < 1:
            return []

        return sorted(
            self._incidents.values(),
            key=lambda incident: incident.updated_at,
            reverse=True,
        )[:limit]

    def reset(self) -> None:
        """Reset all incident state."""
        self._candidates.clear()
        self._incidents.clear()
        self._dedup.clear()
        self._counter = count(1)


# ---------------------------------------------------------------------------
# Default engine
# ---------------------------------------------------------------------------


def build_default_incident_engine() -> IncidentEngine:
    """Build the default incident correlation engine."""
    return IncidentEngine(
        minimum_metrics=2,
        minimum_anomalies=3,
        correlation_window_seconds=180,
        incident_ttl_seconds=600,
        resolution_quiet_seconds=120,
        max_incidents=100,
    )


__all__ = [
    "IncidentEngine",
    "build_default_incident_engine",
]
