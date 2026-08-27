from __future__ import annotations

from collections import defaultdict, deque
from statistics import mean, pstdev
from typing import Deque, Iterable

from .models import DetectedAnomaly, MetricSample


class RollingZScoreDetector:
    """Streaming z-score detector with a rolling window."""

    def __init__(
        self,
        *,
        window_size: int = 16,
        threshold: float = 3.0,
        min_samples: int = 8,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        self._history: dict[tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    @property
    def name(self) -> str:
        return "rolling_zscore"

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        key = (sample.entity, sample.metric)
        history = self._history[key]
        if len(history) < self.min_samples:
            history.append(sample.value)
            return None

        baseline_mean = mean(history)
        baseline_std = pstdev(history) or 1e-6
        z_score = abs(sample.value - baseline_mean) / baseline_std
        history.append(sample.value)

        if z_score < self.threshold:
            return None

        direction = "above" if sample.value > baseline_mean else "below"
        explanation = (
            f"{sample.metric} is {direction} the rolling mean by {z_score:.2f} standard deviations."
        )
        return DetectedAnomaly(
            timestamp=sample.timestamp,
            metric=sample.metric,
            entity=sample.entity,
            value=sample.value,
            expected_mean=baseline_mean,
            expected_std=baseline_std,
            z_score=z_score,
            detector_name=self.name,
            explanation=explanation,
            severity=_severity_from_z(z_score),
        )


class EWMADeviationDetector:
    """Streaming detector that compares values against an EWMA baseline."""

    def __init__(
        self,
        *,
        alpha: float = 0.28,
        threshold: float = 2.0,
        warmup: int = 8,
        min_scale: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.warmup = warmup
        self.min_scale = min_scale
        self._state: dict[tuple[str, str], tuple[float, float, int]] = {}

    @property
    def name(self) -> str:
        return "ewma_deviation"

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        key = (sample.entity, sample.metric)
        state = self._state.get(key)
        if state is None:
            self._state[key] = (sample.value, 0.0, 1)
            return None

        ewma, ewmvar, count = state
        deviation = sample.value - ewma
        sigma = max(ewmvar**0.5, self.min_scale)
        score = abs(deviation) / sigma
        next_ewma = ewma + self.alpha * deviation
        next_ewmvar = (1.0 - self.alpha) * (ewmvar + self.alpha * deviation * deviation)
        self._state[key] = (next_ewma, next_ewmvar, count + 1)

        if count < self.warmup or score < self.threshold:
            return None

        direction = "above" if deviation > 0 else "below"
        explanation = f"{sample.metric} is {direction} the EWMA baseline by {score:.2f} sigma."
        return DetectedAnomaly(
            timestamp=sample.timestamp,
            metric=sample.metric,
            entity=sample.entity,
            value=sample.value,
            expected_mean=ewma,
            expected_std=sigma,
            z_score=score,
            detector_name=self.name,
            explanation=explanation,
            severity=_severity_from_z(score),
        )


class DetectorSuite:
    """Fan-out wrapper for multiple streaming detectors."""

    def __init__(self, detectors: Iterable[object]) -> None:
        self._detectors = list(detectors)

    def observe(self, samples: Iterable[MetricSample]) -> list[DetectedAnomaly]:
        anomalies: list[DetectedAnomaly] = []
        for sample in samples:
            for detector in self._detectors:
                observed = detector.observe(sample)
                if observed is not None:
                    anomalies.append(observed)
        return anomalies


def _severity_from_z(z_score: float) -> str:
    if z_score >= 5.0:
        return "critical"
    if z_score >= 4.0:
        return "high"
    if z_score >= 3.0:
        return "medium"
    return "low"
