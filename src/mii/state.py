from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from .detectors import DetectorSuite, EWMADeviationDetector, RollingZScoreDetector
from .incidents import IncidentEngine
from .graph import DeploymentEvent, DependencyGraph, GraphRCAEngine, RootCauseHypothesis, build_default_dependency_graph
from .monitoring import DataQualityIssue, DataQualityMonitor, DriftFinding, DriftMonitor, FeatureObservation
from .models import DetectedAnomaly, Incident, MetricSample
from .synthetic import SyntheticWorkload


@dataclass(slots=True)
class PhaseOneSnapshot:
    generated_at: datetime
    step: int
    latest_metrics: dict[str, float]
    metric_units: dict[str, str]
    recent_samples: list[MetricSample]
    recent_anomalies: list[DetectedAnomaly]
    recent_feature_observations: list[FeatureObservation]
    drift_findings: list[DriftFinding]
    quality_issues: list[DataQualityIssue]
    deployment_events: list[DeploymentEvent]
    dependency_graph: DependencyGraph
    root_cause_hypotheses: list[RootCauseHypothesis]
    incident: Incident | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "step": self.step,
            "latest_metrics": dict(self.latest_metrics),
            "metric_units": dict(self.metric_units),
            "recent_samples": [sample.to_dict() for sample in self.recent_samples],
            "recent_anomalies": [anomaly.to_dict() for anomaly in self.recent_anomalies],
            "recent_feature_observations": [observation.to_dict() for observation in self.recent_feature_observations],
            "drift_findings": [finding.to_dict() for finding in self.drift_findings],
            "quality_issues": [issue.to_dict() for issue in self.quality_issues],
            "deployment_events": [event.to_dict() for event in self.deployment_events],
            "dependency_graph": self.dependency_graph.to_dict(),
            "root_cause_hypotheses": [hypothesis.to_dict() for hypothesis in self.root_cause_hypotheses],
            "incident": None if self.incident is None else self.incident.to_dict(),
        }


@dataclass(slots=True)
class PhaseOneRuntime:
    """Coordinates the Phase 1 streaming slice."""

    workload: SyntheticWorkload = field(default_factory=SyntheticWorkload)
    detector_suite: DetectorSuite = field(
        default_factory=lambda: DetectorSuite(
            [
                RollingZScoreDetector(),
                EWMADeviationDetector(),
            ]
        )
    )
    incident_engine: IncidentEngine = field(default_factory=IncidentEngine)
    drift_monitor: DriftMonitor = field(default_factory=DriftMonitor)
    quality_monitor: DataQualityMonitor = field(default_factory=DataQualityMonitor)
    dependency_graph: DependencyGraph = field(default_factory=build_default_dependency_graph)
    rca_engine: GraphRCAEngine = field(init=False, repr=False)
    history_size: int = 40
    _deployment_events: list[DeploymentEvent] = field(init=False, repr=False)
    _recent_samples: list[MetricSample] = field(init=False, repr=False)
    _recent_anomalies: list[DetectedAnomaly] = field(init=False, repr=False)
    _recent_feature_observations: list[FeatureObservation] = field(init=False, repr=False)
    _drift_findings: list[DriftFinding] = field(init=False, repr=False)
    _quality_issues: list[DataQualityIssue] = field(init=False, repr=False)
    _root_cause_hypotheses: list[RootCauseHypothesis] = field(init=False, repr=False)
    _latest_metrics: dict[str, float] = field(init=False, repr=False)
    _metric_units: dict[str, str] = field(init=False, repr=False)
    _running: Event = field(init=False, repr=False)
    _worker: Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.rca_engine = GraphRCAEngine(self.dependency_graph)
        self._deployment_events: list[DeploymentEvent] = []
        self._recent_samples: list[MetricSample] = []
        self._recent_anomalies: list[DetectedAnomaly] = []
        self._recent_feature_observations: list[FeatureObservation] = []
        self._drift_findings: list[DriftFinding] = []
        self._quality_issues: list[DataQualityIssue] = []
        self._root_cause_hypotheses: list[RootCauseHypothesis] = []
        self._latest_metrics: dict[str, float] = {}
        self._metric_units: dict[str, str] = {}
        self._running = Event()
        self._worker: Thread | None = None

    @property
    def active_incident(self) -> Incident | None:
        return self.incident_engine.active_incident

    @property
    def step(self) -> int:
        return self.workload.step

    def tick(self) -> PhaseOneSnapshot:
        samples = self.workload.next_frame()
        feature_observations = self.workload.feature_frame()
        deployment_events = self.workload.deployment_events()
        anomalies = self.detector_suite.observe(samples)
        drift_findings = [finding for finding in (self.drift_monitor.observe(observation) for observation in feature_observations) if finding is not None]
        quality_issues = [issue for issue in (self.quality_monitor.observe(observation) for observation in feature_observations) if issue is not None]
        root_cause_hypotheses = self.rca_engine.analyze(
            step=self.step,
            anomalies=anomalies,
            drift_findings=drift_findings,
            quality_issues=quality_issues,
            feature_observations=feature_observations,
            deployment_events=deployment_events,
        )
        incident = self.incident_engine.observe(
            step=self.step,
            anomalies=anomalies,
            timestamp=samples[0].timestamp,
        )
        self._append_samples(samples)
        self._append_anomalies(anomalies)
        self._append_feature_observations(feature_observations)
        self._append_drift_findings(drift_findings)
        self._append_quality_issues(quality_issues)
        self._append_deployment_events(deployment_events)
        self._append_root_cause_hypotheses(root_cause_hypotheses)
        return self.snapshot()

    def _append_samples(self, samples: list[MetricSample]) -> None:
        self._recent_samples.extend(samples)
        self._recent_samples = self._recent_samples[-self.history_size :]
        for sample in samples:
            self._latest_metrics[sample.metric] = sample.value
            self._metric_units[sample.metric] = sample.unit

    def _append_anomalies(self, anomalies: list[DetectedAnomaly]) -> None:
        self._recent_anomalies.extend(anomalies)
        self._recent_anomalies = self._recent_anomalies[-self.history_size :]

    def _append_feature_observations(self, observations: list[FeatureObservation]) -> None:
        self._recent_feature_observations.extend(observations)
        self._recent_feature_observations = self._recent_feature_observations[-self.history_size :]

    def _append_drift_findings(self, findings: list[DriftFinding]) -> None:
        self._drift_findings.extend(findings)
        self._drift_findings = self._drift_findings[-self.history_size :]

    def _append_quality_issues(self, issues: list[DataQualityIssue]) -> None:
        self._quality_issues.extend(issues)
        self._quality_issues = self._quality_issues[-self.history_size :]

    def _append_deployment_events(self, events: list[DeploymentEvent]) -> None:
        self._deployment_events.extend(events)
        self._deployment_events = self._deployment_events[-self.history_size :]

    def _append_root_cause_hypotheses(self, hypotheses: list[RootCauseHypothesis]) -> None:
        self._root_cause_hypotheses = list(hypotheses)

    def snapshot(self) -> PhaseOneSnapshot:
        return PhaseOneSnapshot(
            generated_at=datetime.now(timezone.utc),
            step=self.step,
            latest_metrics=dict(self._latest_metrics),
            metric_units=dict(self._metric_units),
            recent_samples=list(self._recent_samples),
            recent_anomalies=list(self._recent_anomalies),
            recent_feature_observations=list(self._recent_feature_observations),
            drift_findings=list(self._drift_findings),
            quality_issues=list(self._quality_issues),
            deployment_events=list(self._deployment_events),
            dependency_graph=self.dependency_graph,
            root_cause_hypotheses=list(self._root_cause_hypotheses),
            incident=self.active_incident,
        )

    def run_background(self, *, interval_seconds: float = 1.0) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._running.set()
        self._worker = Thread(target=self._run_loop, kwargs={"interval_seconds": interval_seconds}, daemon=True)
        self._worker.start()

    def stop_background(self) -> None:
        self._running.clear()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def _run_loop(self, *, interval_seconds: float) -> None:
        while self._running.is_set():
            self.tick()
            self._running.wait(interval_seconds)
