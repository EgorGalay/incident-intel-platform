from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count

from .models import DetectedAnomaly, Incident


@dataclass(slots=True)
class IncidentEngine:
    """Create incidents from correlated anomalies."""

    minimum_metrics: int = 2
    minimum_anomalies: int = 3
    correlation_window: int = 3
    incident_ttl_seconds: int = 600
    _incident_counter: count = field(default_factory=lambda: count(1427), init=False, repr=False)
    _recent: deque[tuple[int, DetectedAnomaly]] = field(init=False, repr=False)
    _active_incident: Incident | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._recent: deque[tuple[int, DetectedAnomaly]] = deque()
        self._active_incident: Incident | None = None

    @property
    def active_incident(self) -> Incident | None:
        return self._active_incident

    def observe(
        self,
        *,
        step: int,
        anomalies: list[DetectedAnomaly],
        timestamp: datetime,
    ) -> Incident | None:
        for anomaly in anomalies:
            self._recent.append((step, anomaly))

        while self._recent and step - self._recent[0][0] > self.correlation_window:
            self._recent.popleft()

        if self._active_incident is not None:
            return None

        recent_anomalies = [anomaly for _, anomaly in self._recent]
        metrics = {anomaly.metric for anomaly in recent_anomalies}
        if len(recent_anomalies) < self.minimum_anomalies or len(metrics) < self.minimum_metrics:
            return None

        incident = self._create_incident(recent_anomalies, timestamp)
        self._active_incident = incident
        return incident

    def _create_incident(self, anomalies: list[DetectedAnomaly], timestamp: datetime) -> Incident:
        incident_id = f"INC-{next(self._incident_counter)}"
        max_score = max(anomaly.z_score for anomaly in anomalies)
        distinct_metrics = sorted({anomaly.metric for anomaly in anomalies})
        confidence = round(min(0.99, 0.56 + 0.06 * len(distinct_metrics) + 0.05 * max_score), 2)
        severity = "critical" if max_score >= 5.0 or len(distinct_metrics) >= 3 else "high"
        title = "Feature service degradation detected"
        summary = (
            "Correlated anomalies were detected across "
            + ", ".join(distinct_metrics)
            + ". The pattern is consistent with a feature-service degradation."
        )
        actions = [
            "Inspect the feature-service deployment timeline",
            "Check missing-feature and latency telemetry",
            "Validate model inputs before re-enabling full traffic",
        ]
        return Incident(
            incident_id=incident_id,
            title=title,
            detected_at=timestamp.astimezone(timezone.utc),
            severity=severity,
            confidence=confidence,
            summary=summary,
            observed_anomalies=list(anomalies),
            recommended_actions=actions,
        )
