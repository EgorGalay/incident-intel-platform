from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from math import log
from statistics import mean
from typing import Deque


@dataclass(slots=True)
class FeatureObservation:
    """Synthetic per-request feature telemetry for monitoring and drift analysis."""

    timestamp: datetime
    request_id: str
    feature_name: str
    entity: str
    value: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "feature_name": self.feature_name,
            "entity": self.entity,
            "value": self.value,
        }


@dataclass(slots=True)
class DriftFinding:
    """Population drift signal for one feature."""

    timestamp: datetime
    feature_name: str
    psi: float
    baseline_mean: float
    current_mean: float
    explanation: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
            "psi": self.psi,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass(slots=True)
class DataQualityIssue:
    """Data quality issue derived from a feature telemetry window."""

    timestamp: datetime
    feature_name: str
    issue_type: str
    rate: float
    description: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
            "issue_type": self.issue_type,
            "rate": self.rate,
            "description": self.description,
            "severity": self.severity,
        }


class DriftMonitor:
    """Tracks baseline feature distributions and flags PSI-based drift."""

    def __init__(
        self,
        *,
        baseline_size: int = 16,
        window_size: int = 20,
        psi_threshold: float = 0.2,
    ) -> None:
        self.baseline_size = baseline_size
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self._baseline: dict[str, list[float]] = defaultdict(list)
        self._recent: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def observe(self, observation: FeatureObservation) -> DriftFinding | None:
        if observation.value is None:
            return None

        baseline = self._baseline[observation.feature_name]
        if len(baseline) < self.baseline_size:
            baseline.append(observation.value)
            self._recent[observation.feature_name].append(observation.value)
            return None

        recent = self._recent[observation.feature_name]
        recent.append(observation.value)
        if len(recent) < max(6, self.window_size // 3):
            return None

        psi = _psi(baseline, list(recent))
        if psi < self.psi_threshold:
            return None

        baseline_mean = mean(baseline)
        current_mean = mean(recent)
        direction = "higher" if current_mean > baseline_mean else "lower"
        return DriftFinding(
            timestamp=observation.timestamp,
            feature_name=observation.feature_name,
            psi=round(psi, 3),
            baseline_mean=round(baseline_mean, 3),
            current_mean=round(current_mean, 3),
            explanation=f"{observation.feature_name} distribution drifted {direction} than the baseline.",
            severity=_severity_from_psi(psi),
        )


class DataQualityMonitor:
    """Monitors missingness and out-of-range feature values."""

    def __init__(
        self,
        *,
        window_size: int = 20,
        missing_rate_threshold: float = 0.15,
    ) -> None:
        self.window_size = window_size
        self.missing_rate_threshold = missing_rate_threshold
        self._windows: dict[str, Deque[FeatureObservation]] = defaultdict(lambda: deque(maxlen=window_size))
        self._last_seen_request: dict[str, str] = {}

    def observe(self, observation: FeatureObservation) -> DataQualityIssue | None:
        window = self._windows[observation.feature_name]
        window.append(observation)

        if observation.value is None:
            issue = self._maybe_missing_issue(observation.feature_name, observation.timestamp, window)
            if issue is not None:
                return issue

        if observation.feature_name == "user_age" and observation.value is not None:
            if observation.value < 0 or observation.value > 100:
                return DataQualityIssue(
                    timestamp=observation.timestamp,
                    feature_name=observation.feature_name,
                    issue_type="out_of_range",
                    rate=1.0,
                    description="user_age is outside the expected [0, 100] range.",
                    severity="high",
                )

        previous_request = self._last_seen_request.get(observation.feature_name)
        self._last_seen_request[observation.feature_name] = observation.request_id
        if previous_request == observation.request_id:
            return DataQualityIssue(
                timestamp=observation.timestamp,
                feature_name=observation.feature_name,
                issue_type="duplicate_request",
                rate=1.0,
                description=f"Duplicate request telemetry detected for {observation.feature_name}.",
                severity="low",
            )

        return None

    def _maybe_missing_issue(
        self,
        feature_name: str,
        timestamp: datetime,
        window: Deque[FeatureObservation],
    ) -> DataQualityIssue | None:
        if not window:
            return None

        missing = sum(1 for item in window if item.value is None)
        rate = missing / len(window)
        if rate < self.missing_rate_threshold:
            return None

        return DataQualityIssue(
            timestamp=timestamp,
            feature_name=feature_name,
            issue_type="missing_rate",
            rate=round(rate, 3),
            description=f"{feature_name} missing rate is {rate:.1%} over the recent window.",
            severity="high" if rate >= 0.3 else "medium",
        )


def _psi(expected: list[float], observed: list[float], bins: int = 8) -> float:
    low = min(min(expected), min(observed))
    high = max(max(expected), max(observed))
    if high == low:
        return 0.0

    step = (high - low) / bins
    boundaries = [low + step * idx for idx in range(1, bins)]
    expected_hist = _histogram(expected, boundaries, bins)
    observed_hist = _histogram(observed, boundaries, bins)

    score = 0.0
    for exp, obs in zip(expected_hist, observed_hist, strict=True):
        score += (obs - exp) * log(obs / exp)
    return score


def _histogram(values: list[float], boundaries: list[float], bins: int) -> list[float]:
    counts = [0] * bins
    for value in values:
        index = 0
        while index < len(boundaries) and value > boundaries[index]:
            index += 1
        counts[index] += 1

    total = sum(counts) or 1
    return [max(count / total, 1e-6) for count in counts]


def _severity_from_psi(psi: float) -> str:
    if psi >= 0.5:
        return "high"
    if psi >= 0.3:
        return "medium"
    return "low"
