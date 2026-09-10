
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Deque, Iterable

from .models import (
    DetectedAnomaly,
    MetricSample,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _severity_from_z(z_score: float) -> str:
    """
    Convert absolute z-score magnitude into an incident-friendly severity.

    The detector itself does not decide incident lifecycle. It only provides
    a local severity signal.
    """
    magnitude = abs(z_score)

    if magnitude >= 5.0:
        return SEVERITY_CRITICAL

    if magnitude >= 4.0:
        return SEVERITY_HIGH

    if magnitude >= 3.0:
        return SEVERITY_MEDIUM

    return SEVERITY_LOW


def _direction(value: float, baseline: float) -> str:
    """Return human-readable direction relative to a baseline."""
    if value > baseline:
        return "above"

    if value < baseline:
        return "below"

    return "at"


def _anomaly_message(
    *,
    metric: str,
    value: float,
    baseline: float,
    z_score: float,
    detector: str,
) -> str:
    """Build a deterministic anomaly explanation."""
    direction = _direction(value, baseline)

    if direction == "at":
        return (
            f"{metric} is at the {detector} baseline "
            f"with z-score {z_score:.2f}."
        )

    return (
        f"{metric} is {direction} the {detector} baseline "
        f"by {abs(z_score):.2f} standard deviations."
    )


# ---------------------------------------------------------------------------
# EWMA anomaly detector
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _EWMAState:
    """Internal per-series EWMA state."""

    mean: float
    variance: float
    count: int


class EWMAAnomalyDetector:
    """
    Streaming EWMA anomaly detector.

    The detector evaluates the current sample against the state accumulated
    from previous samples. The current sample is only added to the state
    after detection, preventing the observation from contaminating its own
    baseline.

    Parameters
    ----------
    alpha:
        EWMA smoothing factor. Higher values react faster to changes.

    threshold:
        Absolute z-score required to emit an anomaly.

    min_history:
        Number of previous observations required before anomaly detection
        begins.

    min_scale:
        Minimum standard deviation used for numerical stability when the
        observed series has very low variance.

    Notes
    -----
    z_score is signed:

        positive -> value is above baseline
        negative -> value is below baseline
    """

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        threshold: float = 3.0,
        min_history: int = 5,
        min_scale: float = 1e-6,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1].")

        if threshold <= 0:
            raise ValueError("threshold must be greater than zero.")

        if min_history < 1:
            raise ValueError("min_history must be at least 1.")

        if min_scale < 0:
            raise ValueError("min_scale cannot be negative.")

        self.alpha = alpha
        self.threshold = threshold
        self.min_history = min_history
        self.min_scale = min_scale

        self._state: dict[tuple[str, str], _EWMAState] = {}

    @property
    def name(self) -> str:
        """Stable detector identifier."""
        return "ewma"

    def update(self, sample: MetricSample) -> None:
        """
        Update the detector state with a sample.

        ``update`` does not perform detection. This separation makes it
        possible to evaluate a sample against the previous baseline first.
        """
        key = (sample.entity, sample.metric)
        state = self._state.get(key)

        if state is None:
            self._state[key] = _EWMAState(
                mean=sample.value,
                variance=0.0,
                count=1,
            )
            return

        deviation = sample.value - state.mean

        next_mean = (
            state.mean
            + self.alpha * deviation
        )

        next_variance = (
            (1.0 - self.alpha)
            * (
                state.variance
                + self.alpha * deviation * deviation
            )
        )

        self._state[key] = _EWMAState(
            mean=next_mean,
            variance=max(0.0, next_variance),
            count=state.count + 1,
        )

    def detect(self, sample: MetricSample) -> DetectedAnomaly | None:
        """
        Detect an anomaly against the previously observed history.

        The sample itself is not added to the detector state.
        """
        key = (sample.entity, sample.metric)
        state = self._state.get(key)

        if state is None:
            return None

        if state.count < self.min_history:
            return None

        baseline = state.mean
        sigma = sqrt(max(state.variance, 0.0))

        # Constant/near-constant series need special handling.
        if sigma < self.min_scale:
            if abs(sample.value - baseline) < self.min_scale:
                return None

            # A non-zero deviation from a truly constant baseline is
            # treated as an extremely strong anomaly.
            z_score = (
                float("inf")
                if sample.value > baseline
                else float("-inf")
            )
        else:
            z_score = (sample.value - baseline) / sigma

        if abs(z_score) < self.threshold:
            return None

        severity = _severity_from_z(z_score)

        return DetectedAnomaly(
            metric=sample.metric,
            entity=sample.entity,
            value=sample.value,
            baseline=baseline,
            z_score=z_score,
            severity=severity,
            timestamp=sample.timestamp,
            detector=self.name,
            message=_anomaly_message(
                metric=sample.metric,
                value=sample.value,
                baseline=baseline,
                z_score=z_score,
                detector="EWMA",
            ),
        )

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        """
        Backward-compatible one-call API.

        Detection is performed before the sample is incorporated into the
        baseline.
        """
        anomaly = self.detect(sample)
        self.update(sample)
        return anomaly

    def reset(self) -> None:
        """Clear all detector state."""
        self._state.clear()


# ---------------------------------------------------------------------------
# Rolling z-score detector
# ---------------------------------------------------------------------------


class RollingZScoreDetector:
    """
    Rolling-window z-score detector.

    This is retained as a compatibility detector for the existing Phase 1-4
    implementation while using the new canonical DetectedAnomaly model.
    """

    def __init__(
        self,
        *,
        window_size: int = 16,
        threshold: float = 3.0,
        min_samples: int = 8,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")

        if threshold <= 0:
            raise ValueError("threshold must be greater than zero.")

        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")

        if min_samples > window_size:
            raise ValueError(
                "min_samples cannot be greater than window_size."
            )

        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples

        self._history: dict[
            tuple[str, str],
            Deque[float],
        ] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    @property
    def name(self) -> str:
        return "rolling_zscore"

    def update(self, sample: MetricSample) -> None:
        """Append a sample to the rolling history."""
        key = (sample.entity, sample.metric)
        self._history[key].append(sample.value)

    def detect(self, sample: MetricSample) -> DetectedAnomaly | None:
        """
        Evaluate the sample against the previous rolling history.

        The sample is not added to the history here.
        """
        key = (sample.entity, sample.metric)
        history = self._history[key]

        if len(history) < self.min_samples:
            return None

        baseline_mean = mean(history)

        baseline_std = pstdev(history)

        if baseline_std == 0.0:
            if sample.value == baseline_mean:
                return None

            z_score = (
                float("inf")
                if sample.value > baseline_mean
                else float("-inf")
            )
        else:
            # IMPORTANT:
            # Preserve the sign. The previous implementation used abs(...)
            # here and therefore lost direction information.
            z_score = (
                (sample.value - baseline_mean)
                / baseline_std
            )

        if abs(z_score) < self.threshold:
            return None

        severity = _severity_from_z(z_score)

        return DetectedAnomaly(
            metric=sample.metric,
            entity=sample.entity,
            value=sample.value,
            baseline=baseline_mean,
            z_score=z_score,
            severity=severity,
            timestamp=sample.timestamp,
            detector=self.name,
            message=_anomaly_message(
                metric=sample.metric,
                value=sample.value,
                baseline=baseline_mean,
                z_score=z_score,
                detector="rolling",
            ),
        )

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        """Backward-compatible detect-then-update operation."""
        anomaly = self.detect(sample)
        self.update(sample)
        return anomaly

    def reset(self) -> None:
        """Clear all rolling histories."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Backward-compatible EWMA detector name
# ---------------------------------------------------------------------------


class EWMADeviationDetector(EWMAAnomalyDetector):
    """
    Backward-compatible alias for the previous detector.

    Existing code may still instantiate:

        EWMADeviationDetector(
            alpha=0.28,
            threshold=2.0,
            warmup=8,
        )

    The new canonical API is ``EWMAAnomalyDetector``.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.28,
        threshold: float = 2.0,
        warmup: int = 8,
        min_scale: float = 0.05,
    ) -> None:
        super().__init__(
            alpha=alpha,
            threshold=threshold,
            min_history=warmup,
            min_scale=min_scale,
        )

    @property
    def name(self) -> str:
        return "ewma_deviation"


# ---------------------------------------------------------------------------
# Data quality detector
# ---------------------------------------------------------------------------


class DataQualityDetector:
    """
    Detect obvious data-quality violations in canonical ML telemetry.

    Supported metrics:

        feature_missing_rate
        error_rate
        traffic_rps
        prediction_latency_ms
        prediction_ctr

    The detector is intentionally deterministic and does not require a
    learned baseline.

    Thresholds can be overridden at construction time.
    """

    DEFAULT_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
        # (minimum_allowed, maximum_allowed)
        "feature_missing_rate": (None, 0.20),
        "error_rate": (None, 0.05),
        "traffic_rps": (10.0, None),
        "prediction_latency_ms": (None, 1000.0),
        "prediction_ctr": (0.01, 0.50),
    }

    def __init__(
        self,
        *,
        thresholds: dict[
            str,
            tuple[float | None, float | None],
        ] | None = None,
    ) -> None:
        self.thresholds = dict(
            thresholds
            if thresholds is not None
            else self.DEFAULT_THRESHOLDS
        )

    @property
    def name(self) -> str:
        return "data_quality"

    def update(self, sample: MetricSample) -> None:
        """Data-quality detection is stateless, so update is a no-op."""
        return None

    def detect(self, sample: MetricSample) -> DetectedAnomaly | None:
        """Return an anomaly when the metric violates its configured range."""
        limits = self.thresholds.get(sample.metric)

        if limits is None:
            return None

        minimum, maximum = limits

        if minimum is not None and sample.value < minimum:
            baseline = minimum

            # Use a normalized score instead of pretending this is a true
            # statistical z-score.
            denominator = max(abs(minimum), 1e-6)
            z_score = (
                (sample.value - baseline)
                / denominator
            )

            return DetectedAnomaly(
                metric=sample.metric,
                entity=sample.entity,
                value=sample.value,
                baseline=baseline,
                z_score=z_score,
                severity=SEVERITY_HIGH,
                timestamp=sample.timestamp,
                detector=self.name,
                message=(
                    f"{sample.metric} dropped below the allowed minimum "
                    f"of {minimum:.4f}; observed {sample.value:.4f}."
                ),
            )

        if maximum is not None and sample.value > maximum:
            baseline = maximum
            denominator = max(abs(maximum), 1e-6)
            z_score = (
                (sample.value - baseline)
                / denominator
            )

            severity = (
                SEVERITY_CRITICAL
                if sample.value > maximum * 2.0
                else SEVERITY_HIGH
            )

            return DetectedAnomaly(
                metric=sample.metric,
                entity=sample.entity,
                value=sample.value,
                baseline=baseline,
                z_score=z_score,
                severity=severity,
                timestamp=sample.timestamp,
                detector=self.name,
                message=(
                    f"{sample.metric} exceeded the allowed maximum "
                    f"of {maximum:.4f}; observed {sample.value:.4f}."
                ),
            )

        return None

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        """Compatibility API."""
        return self.detect(sample)

    def reset(self) -> None:
        """Reset detector state; currently a no-op because it is stateless."""
        return None


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _DriftState:
    """Internal state for one metric/entity drift series."""

    history: Deque[float]


class DriftDetector:
    """
    Lightweight rolling drift detector.

    Drift is detected when the latest value moves sufficiently far from the
    historical rolling baseline.

    This detector is deliberately simple: it is intended as a streaming
    signal that can later be combined with feature-level PSI/KS detectors.
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        threshold: float = 2.5,
        min_history: int = 10,
        min_scale: float = 1e-6,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")

        if threshold <= 0:
            raise ValueError("threshold must be greater than zero.")

        if min_history < 2:
            raise ValueError("min_history must be at least 2.")

        if min_history > window_size:
            raise ValueError(
                "min_history cannot be greater than window_size."
            )

        self.window_size = window_size
        self.threshold = threshold
        self.min_history = min_history
        self.min_scale = min_scale

        self._state: dict[
            tuple[str, str],
            _DriftState,
        ] = {}

    @property
    def name(self) -> str:
        return "drift"

    def update(self, sample: MetricSample) -> None:
        """Add a sample to the drift baseline."""
        key = (sample.entity, sample.metric)

        state = self._state.get(key)

        if state is None:
            state = _DriftState(
                history=deque(maxlen=self.window_size)
            )
            self._state[key] = state

        state.history.append(sample.value)

    def detect(self, sample: MetricSample) -> DetectedAnomaly | None:
        """Detect a statistically significant shift from the rolling baseline."""
        key = (sample.entity, sample.metric)
        state = self._state.get(key)

        if state is None:
            return None

        history = state.history

        if len(history) < self.min_history:
            return None

        baseline = mean(history)
        scale = pstdev(history)

        if scale < self.min_scale:
            if abs(sample.value - baseline) < self.min_scale:
                return None

            z_score = (
                float("inf")
                if sample.value > baseline
                else float("-inf")
            )
        else:
            z_score = (sample.value - baseline) / scale

        if abs(z_score) < self.threshold:
            return None

        return DetectedAnomaly(
            metric=sample.metric,
            entity=sample.entity,
            value=sample.value,
            baseline=baseline,
            z_score=z_score,
            severity=_severity_from_z(z_score),
            timestamp=sample.timestamp,
            detector=self.name,
            message=_anomaly_message(
                metric=sample.metric,
                value=sample.value,
                baseline=baseline,
                z_score=z_score,
                detector="drift",
            ),
        )

    def observe(self, sample: MetricSample) -> DetectedAnomaly | None:
        """Compatibility detect-then-update API."""
        anomaly = self.detect(sample)
        self.update(sample)
        return anomaly

    def reset(self) -> None:
        """Clear all drift state."""
        self._state.clear()


# ---------------------------------------------------------------------------
# Detector suite
# ---------------------------------------------------------------------------


class DetectorSuite:
    """
    Fan-out wrapper for streaming detectors.

    Every detector receives every sample. Detectors remain independent:
    their internal baselines and lifecycle state are not shared.
    """

    def __init__(self, detectors: Iterable[object]) -> None:
        self._detectors = list(detectors)

    @property
    def detectors(self) -> tuple[object, ...]:
        """Return configured detectors as an immutable view."""
        return tuple(self._detectors)

    def observe(
        self,
        samples: Iterable[MetricSample],
    ) -> list[DetectedAnomaly]:
        """
        Process samples through every configured detector.

        Detectors implementing ``observe`` are preferred. For the canonical
        API, detectors are expected to expose ``detect`` and ``update``.
        """
        anomalies: list[DetectedAnomaly] = []

        for sample in samples:
            for detector in self._detectors:
                observe = getattr(detector, "observe", None)

                if callable(observe):
                    anomaly = observe(sample)
                else:
                    detect = getattr(detector, "detect")
                    update = getattr(detector, "update")

                    anomaly = detect(sample)
                    update(sample)

                if anomaly is not None:
                    anomalies.append(anomaly)

        return anomalies

    def reset(self) -> None:
        """Reset every configured detector."""
        for detector in self._detectors:
            reset = getattr(detector, "reset", None)

            if callable(reset):
                reset()


# ---------------------------------------------------------------------------
# Default detector configuration
# ---------------------------------------------------------------------------


def build_default_detector_suite() -> DetectorSuite:
    """
    Build the default detector stack used by the runtime.

    The stack intentionally combines:

    - EWMA statistical detection
    - rolling z-score compatibility detection
    - explicit data-quality rules
    - rolling drift detection
    """
    return DetectorSuite(
        [
            EWMAAnomalyDetector(
                alpha=0.3,
                threshold=3.0,
                min_history=5,
            ),
            DataQualityDetector(),
            DriftDetector(
                window_size=20,
                threshold=2.5,
                min_history=10,
            ),
        ]
    )


__all__ = [
    "EWMAAnomalyDetector",
    "EWMADeviationDetector",
    "RollingZScoreDetector",
    "DataQualityDetector",
    "DriftDetector",
    "DetectorSuite",
    "build_default_detector_suite",
]

