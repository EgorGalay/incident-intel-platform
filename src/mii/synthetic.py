from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from random import Random

from .graph import DeploymentEvent
from .monitoring import FeatureObservation
from .models import MetricSample, utcnow


@dataclass(slots=True)
class FaultProfile:
    """Deterministic synthetic fault injected into the workload."""

    trigger_step: int = 20
    ctr_drop: float = 0.18
    latency_spike_ms: float = 240.0
    missing_rate_increase: float = 0.29
    error_rate_increase: float = 0.072


@dataclass(slots=True)
class FeatureProfile:
    """Synthetic feature drift and quality degradation settings."""

    drift_trigger_step: int = 16
    quality_trigger_step: int = 22
    deployment_trigger_step: int = 18
    user_age_shift: float = 11.0
    missing_probability: float = 0.24


class SyntheticWorkload:
    """Produces phase 1 telemetry for one ML serving service."""

    def __init__(
        self,
        *,
        seed: int = 7,
        fault_profile: FaultProfile | None = None,
        feature_profile: FeatureProfile | None = None,
    ) -> None:
        self._rng = Random(seed)
        self._fault_profile = fault_profile or FaultProfile()
        self._feature_profile = feature_profile or FeatureProfile()
        self._step = 0
        self._request_counter = 0
        self._deployment_emitted = False

    @property
    def step(self) -> int:
        return self._step

    def _base_signal(self) -> dict[str, float]:
        noise = self._rng
        return {
            "prediction_ctr": max(0.0, min(1.0, 0.31 + noise.gauss(0.0, 0.008))),
            "prediction_latency_ms": max(1.0, 92.0 + noise.gauss(0.0, 6.0)),
            "feature_missing_rate": max(0.0, 0.021 + noise.gauss(0.0, 0.002)),
            "error_rate": max(0.0, 0.008 + noise.gauss(0.0, 0.0015)),
            "traffic_rps": max(1.0, 225.0 + noise.gauss(0.0, 11.0)),
        }

    def _apply_fault(self, signal: dict[str, float]) -> dict[str, float]:
        profile = self._fault_profile
        degraded = dict(signal)
        degraded["prediction_ctr"] = max(0.0, degraded["prediction_ctr"] - profile.ctr_drop)
        degraded["prediction_latency_ms"] = degraded["prediction_latency_ms"] + profile.latency_spike_ms
        degraded["feature_missing_rate"] = min(1.0, degraded["feature_missing_rate"] + profile.missing_rate_increase)
        degraded["error_rate"] = min(1.0, degraded["error_rate"] + profile.error_rate_increase)
        return degraded

    def next_frame(self) -> list[MetricSample]:
        """Advance the workload by one step and emit telemetry samples."""

        self._step += 1
        base_signal = self._base_signal()
        if self._step >= self._fault_profile.trigger_step:
            base_signal = self._apply_fault(base_signal)

        timestamp = utcnow()
        entity = "feature-service"
        samples = [
            MetricSample(timestamp, "prediction_ctr", entity, base_signal["prediction_ctr"], unit="ratio"),
            MetricSample(timestamp, "prediction_latency_ms", entity, base_signal["prediction_latency_ms"], unit="ms"),
            MetricSample(timestamp, "feature_missing_rate", entity, base_signal["feature_missing_rate"], unit="ratio"),
            MetricSample(timestamp, "error_rate", entity, base_signal["error_rate"], unit="ratio"),
            MetricSample(timestamp, "traffic_rps", entity, base_signal["traffic_rps"], unit="rps"),
        ]
        return samples

    def preview_faulted_values(self) -> dict[str, float]:
        """Expose the deterministic fault profile for tests and docs."""

        signal = self._apply_fault(
            {
                "prediction_ctr": 0.31,
                "prediction_latency_ms": 92.0,
                "feature_missing_rate": 0.021,
                "error_rate": 0.008,
                "traffic_rps": 225.0,
            }
        )
        return signal

    def feature_frame(self) -> list[FeatureObservation]:
        """Generate feature observations used by Phase 2 drift and quality monitoring."""

        self._request_counter += 1
        timestamp = utcnow()
        return [
            self._feature_observation(
                timestamp=timestamp,
                request_id=f"req-{self._request_counter:05d}",
                feature_name="user_age",
                base_value=33.0,
                stddev=3.6,
                lower=18.0,
                upper=75.0,
            ),
            self._feature_observation(
                timestamp=timestamp,
                request_id=f"req-{self._request_counter:05d}",
                feature_name="session_duration_sec",
                base_value=146.0,
                stddev=18.0,
                lower=2.0,
                upper=900.0,
            ),
            self._feature_observation(
                timestamp=timestamp,
                request_id=f"req-{self._request_counter:05d}",
                feature_name="pages_per_session",
                base_value=4.2,
                stddev=0.8,
                lower=1.0,
                upper=18.0,
            ),
        ]

    def deployment_events(self) -> list[DeploymentEvent]:
        """Emit the feature-service deployment that precedes the incident."""

        if self._deployment_emitted or self._step < self._feature_profile.deployment_trigger_step:
            return []

        self._deployment_emitted = True
        timestamp = utcnow() - timedelta(minutes=4)
        return [
            DeploymentEvent(
                timestamp=timestamp,
                component="feature-service",
                version="v2.8",
                step=self._step,
                description="feature-service:v2.8 deployed with feature normalization changes.",
            )
        ]

    def _feature_observation(
        self,
        *,
        timestamp: datetime,
        request_id: str,
        feature_name: str,
        base_value: float,
        stddev: float,
        lower: float,
        upper: float,
    ) -> FeatureObservation:
        value = max(lower, min(upper, base_value + self._rng.gauss(0.0, stddev)))
        if self._step >= self._feature_profile.drift_trigger_step and feature_name == "user_age":
            value = max(lower, min(upper, value + self._feature_profile.user_age_shift))

        if self._step >= self._feature_profile.quality_trigger_step and feature_name == "user_age":
            if self._rng.random() < self._feature_profile.missing_probability:
                return FeatureObservation(timestamp, request_id, feature_name, "feature-service", None)

        return FeatureObservation(timestamp, request_id, feature_name, "feature-service", value)
