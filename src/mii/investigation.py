from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import DetectedAnomaly, Incident, InvestigationReport

logger = logging.getLogger(__name__)


@dataclass
class HistoricalIncident:
    incident_id: str
    title: str
    component: str
    root_cause: str
    severity: str = "medium"
    started_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_report(
            cls,
            report: InvestigationReport,
    ) -> "HistoricalIncident":
        """Create a historical incident from an investigation report."""
        return cls(
            incident_id=report.incident_id,
            title=report.summary or report.verdict or "Historical incident",
            component="unknown",
            root_cause=report.root_cause_analysis or report.verdict or "unknown",
            severity="medium",
            started_at=None,
            tags=[report.mode] if report.mode else [],
            summary=report.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "component": self.component,
            "root_cause": self.root_cause,
            "severity": self.severity,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "tags": list(self.tags),
            "summary": self.summary,
        }


@dataclass
class EvidenceItem:
    source: str
    signal: str
    value: Any
    interpretation: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolTrace:
    tool: str
    input: dict[str, Any]
    output: Any
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentUsage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UsageInfo(dict):
    """
    Dict-compatible usage payload that also supports attribute access.

    ``InvestigationReport.usage`` is a plain ``dict`` in the canonical
    model (so ``to_dict()``/JSON serialization keeps working unchanged),
    but the previous Phase 4 callers/tests access it as
    ``usage.provider`` / ``usage.model`` / ``usage.input_tokens``. This
    class satisfies both without duplicating state.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _estimate_tokens(text: str) -> int:
    """Deterministic, dependency-free token estimate for usage accounting."""
    return max(1, len(str(text).split()))


def build_default_historical_incidents() -> list[HistoricalIncident]:
    return [
        HistoricalIncident(
            "INC-1204",
            "Feature service deployment regression",
            "feature-service",
            "Recent feature-service deployment introduced latency regression",
            "high",
            tags=["deployment", "latency", "feature-service"],
            summary="Latency increased shortly after a feature-service deployment.",
        ),
        HistoricalIncident(
            "INC-1268",
            "Delayed feature pipeline batch",
            "data-pipeline",
            "Delayed upstream data pipeline batch",
            "medium",
            tags=["data-pipeline", "freshness", "features"],
            summary="A delayed batch caused stale features downstream.",
        ),
        HistoricalIncident(
            "INC-1331",
            "API gateway saturation",
            "api-gateway",
            "Gateway saturation caused elevated request latency and errors",
            "high",
            tags=["api-gateway", "traffic", "errors"],
            summary="Traffic saturation resulted in latency and elevated error rate.",
        ),
    ]


class InvestigationService:
    """Grounded incident investigation with Ollama and deterministic fallback."""

    def __init__(
        self,
        config: Any | None = None,
        provider: str | None = None,
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.provider = (
            provider
            or getattr(config, "llm_provider", None)
            or os.getenv("MII_LLM_PROVIDER", "ollama")
        ).strip().lower()
        self.ollama_host = (
            ollama_host
            or getattr(config, "ollama_host", None)
            or os.getenv("MII_OLLAMA_HOST", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.ollama_model = (
            ollama_model
            or getattr(config, "ollama_model", None)
            or os.getenv("MII_OLLAMA_MODEL", "llama3.1")
        )
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(config, "request_timeout", None)
            or os.getenv("MII_OLLAMA_TIMEOUT_SECONDS", "30.0")
        )

    def investigate(
        self,
        incident: Incident,
        rca_result: Any,
        state: Mapping[str, Any] | Any,
    ) -> InvestigationReport:
        prompt = self.build_prompt(incident, rca_result, state)
        if self.provider in {"local", "local-fallback", "fallback", "none"}:
            return self.local_fallback(incident, rca_result, prompt)

        if self.provider == "ollama":
            try:
                text = self.call_ollama(prompt)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty Ollama response")
                return self._make_report(incident, rca_result, text, "ollama", prompt)
            except Exception:
                logger.exception("Ollama investigation failed; using local fallback")
                return self.local_fallback(incident, rca_result, prompt)

        if self.provider == "openai":
            return self.local_fallback(incident, rca_result, prompt)

        return self.local_fallback(incident, rca_result, prompt)

    def build_prompt(
        self,
        incident: Incident,
        rca_result: Any,
        state: Mapping[str, Any] | Any,
    ) -> str:
        payload = {
            "incident": self._incident_dict(incident),
            "rca": self._to_dict(rca_result),
            "state": self._state_context(state),
        }
        return (
            "You are an ML incident investigation assistant.\n"
            "Use ONLY supplied evidence. Do not invent facts, logs, metrics, "
            "deployments, or dependencies.\n"
            "Explain the likely root cause, evidence, downstream effects, "
            "and recommended actions.\n\n"
            + json.dumps(payload, default=str, ensure_ascii=False, indent=2)
        )

    def call_ollama(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.ollama_model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = Request(
            f"{self.ollama_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Ollama request failed") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama response has no usable text")
        return self._sanitize(text)

    def local_fallback(
        self,
        incident: Incident,
        rca_result: Any,
        prompt: str = "",
    ) -> InvestigationReport:
        root = self._value(rca_result, "root_cause") or self._value(rca_result, "root_cause_label")
        confidence = self._confidence(rca_result)
        evidence = self._evidence_strings(rca_result)
        downstream = self._list_value(rca_result, "downstream_effects")
        recommendations = list(incident.recommendations)
        if not recommendations:
            recommendations = [
                "Validate the suspected root component.",
                "Check recent deployments and configuration changes.",
                "Inspect upstream and downstream dependency health.",
                "Continue monitoring affected metrics after remediation.",
            ]
        verdict = (
            f"Root cause: {root}."
            if root
            else "Root cause is not determined from available evidence."
        )
        report = InvestigationReport(
            incident_id=incident.incident_id,
            mode="local-fallback",
            verdict=verdict,
            confidence=confidence,
            summary=(
                "Grounded investigation based on deterministic telemetry, "
                "RCA evidence, and dependency relationships."
            ),
            root_cause_analysis=verdict,
            evidence=evidence,
            downstream_effects=downstream,
            recommendations=recommendations,
            historical_matches=self._historical_strings(rca_result),
            usage=UsageInfo(
                provider="local-fallback",
                model="deterministic",
                prompt_tokens=_estimate_tokens(prompt),
                completion_tokens=0,
                input_tokens=_estimate_tokens(prompt),
                output_tokens=0,
                total_tokens=_estimate_tokens(prompt),
                estimated_cost_usd=0.0,
            ),
        )
        self._attach_compatibility(report, "local-fallback")
        return report

    def _make_report(
        self,
        incident: Incident,
        rca_result: Any,
        text: str,
        mode: str,
        prompt: str = "",
    ) -> InvestigationReport:
        input_tokens = _estimate_tokens(prompt)
        output_tokens = _estimate_tokens(text)
        report = InvestigationReport(
            incident_id=incident.incident_id,
            mode=mode,
            verdict="LLM-assisted investigation completed.",
            confidence=self._confidence(rca_result),
            summary=self._sanitize(text),
            root_cause_analysis=str(
                self._value(rca_result, "root_cause")
                or "See deterministic RCA evidence."
            ),
            evidence=self._evidence_strings(rca_result),
            downstream_effects=self._list_value(rca_result, "downstream_effects"),
            recommendations=list(incident.recommendations),
            historical_matches=self._historical_strings(rca_result),
            usage=UsageInfo(
                provider=mode,
                model=self.ollama_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                estimated_cost_usd=0.0,
            ),
        )
        self._attach_compatibility(report, mode)
        return report

    @staticmethod
    def _attach_compatibility(report: InvestigationReport, mode: str) -> None:
        """Keep attributes used by the previous Phase 4 tests/callers."""
        report.tool_trace = [
            ToolTrace(
                tool="deterministic-rca",
                input={},
                output=report.root_cause_analysis,
            ).to_dict()
        ]
        report.usage.setdefault("provider", mode)

    @staticmethod
    def _sanitize(text: str) -> str:
        text = str(text).strip()
        return text[:12000] + ("\n[output truncated]" if len(text) > 12000 else "")

    @staticmethod
    def _incident_dict(incident: Incident) -> dict[str, Any]:
        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity,
            "status": incident.status,
            "component": incident.component,
            "root_cause": incident.root_cause,
            "confidence": incident.confidence,
            "affected_components": incident.affected_components,
            "anomalies": [
                {
                    "metric": a.metric,
                    "entity": a.entity,
                    "value": a.value,
                    "baseline": a.baseline,
                    "z_score": a.z_score,
                    "severity": a.severity,
                    "detector": a.detector,
                    "message": a.message,
                }
                for a in incident.anomalies
            ],
        }

    @staticmethod
    def _state_context(state: Mapping[str, Any] | Any) -> Any:
        if isinstance(state, Mapping):
            return {
                key: state[key]
                for key in (
                    "step",
                    "latest_metrics",
                    "recent_anomalies",
                    "root_cause_hypotheses",
                )
                if key in state
            }
        return {
            "step": getattr(state, "step", None),
            "latest_metrics": getattr(state, "latest_metrics", {}),
            "recent_anomalies": getattr(state, "recent_anomalies", []),
            "root_cause_hypotheses": getattr(state, "root_cause_hypotheses", []),
        }

    @staticmethod
    def _to_dict(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [InvestigationService._to_dict(v) for v in value]
        return str(value)

    @staticmethod
    def _value(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    @classmethod
    def _confidence(cls, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(cls._value(value, "confidence", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _list_value(cls, value: Any, key: str) -> list[str]:
        raw = cls._value(value, key, []) or []
        return [str(x) for x in raw]

    @classmethod
    def _evidence_strings(cls, value: Any) -> list[str]:
        raw = cls._value(value, "evidence", []) or []
        result: list[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(str(item.get("interpretation") or item.get("signal") or item))
            else:
                result.append(str(item))
        return result

    @classmethod
    def _historical_strings(cls, value: Any) -> list[str]:
        raw = cls._value(value, "historical_matches", []) or []
        return [
            str(item if isinstance(item, str) else cls._value(item, "incident_id", item))
            for item in raw
        ]


class InvestigationToolbelt:
    def __init__(self, state: Mapping[str, Any] | Any = None) -> None:
        self.state = state

    def collect_recent_metrics(self, state: Mapping[str, Any] | Any | None = None) -> list[Any]:
        source = state if state is not None else self.state
        if source is None:
            return []
        if isinstance(source, Mapping):
            return list(source.get("recent_samples", []))
        return list(getattr(source, "recent_samples", []))

    def collect_anomalies(self, incident: Incident) -> list[DetectedAnomaly]:
        return list(incident.anomalies)

    def collect_dependency_context(self, rca_result: Any) -> dict[str, Any]:
        return {
            "root_cause": InvestigationService._value(rca_result, "root_cause"),
            "downstream_effects": InvestigationService._list_value(rca_result, "downstream_effects"),
        }


class IncidentInvestigator:
    """
    Compatibility wrapper for the previous Phase 4 runtime.

    On top of ``InvestigationService`` (which only knows the canonical
    ``local-fallback`` / ``ollama`` / ``openai`` modes), this wrapper
    restores the previous Phase 4 mode vocabulary and decision rule:

        no_llm=True
            -> always "local-fallback" (LLM disabled by configuration).
        llm_provider given explicitly
            -> always attempt that provider; canonical "ollama"/"openai"
               are reported back as "llm", canonical "local-fallback"
               (i.e. the attempt failed) stays "local-fallback".
        llm_provider omitted (auto mode)
            -> if the deterministic RCA/incident confidence already meets
               ``llm_confidence_threshold``, the LLM is not even called
               ("ml-sufficient"); this also avoids unnecessary network
               calls (and the timeouts that come with them) when the
               deterministic evidence is already strong enough.
            -> otherwise the configured provider (env/config default) is
               attempted, same remapping as above.
    """

    def __init__(
            self,
            llm_provider: str | None = None,
            ollama_host: str | None = None,
            ollama_model: str | None = None,
            timeout_seconds: float | None = None,
            no_llm: bool = False,
            llm_confidence_threshold: float = 0.65,
    ) -> None:
        self.llm_confidence_threshold = llm_confidence_threshold
        self.no_llm = no_llm
        self._explicit_provider = llm_provider is not None

        self.service = InvestigationService(
            provider="local-fallback" if no_llm else llm_provider,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            timeout_seconds=timeout_seconds,
        )

    def investigate(
        self,
        snapshot: Any,
        historical_incidents: Sequence[Any] | None = None,
    ) -> InvestigationReport | None:
        incident = getattr(snapshot, "incident", None)
        if incident is None and isinstance(snapshot, Mapping):
            incident = snapshot.get("incident")
        if incident is None:
            return None

        rca_result = getattr(snapshot, "rca_result", None)
        if rca_result is None:
            rca_result = getattr(snapshot, "root_cause_result", None)
        if rca_result is None:
            hypotheses = getattr(snapshot, "root_cause_hypotheses", [])
            first = hypotheses[0] if hypotheses else None
            evidence = list(getattr(first, "evidence_signals", []) or []) if first else []
            rca_result = {
                "root_cause": getattr(first, "component", None) if first else incident.root_cause,
                "confidence": getattr(first, "confidence", incident.confidence) if first else incident.confidence,
                "evidence": evidence,
                "downstream_effects": getattr(first, "downstream_effects", None) or incident.affected_components,
                "historical_matches": [getattr(x, "incident_id", str(x)) for x in (historical_incidents or [])[:5]],
            }

        if self.no_llm:
            report = self.service.investigate(incident, rca_result, snapshot)
            report.skipped_reason = "LLM disabled (no_llm=True)."
            return report

        if not self._explicit_provider:
            confidence = InvestigationService._confidence(rca_result)
            if confidence >= self.llm_confidence_threshold:
                report = self.service.local_fallback(
                    incident, rca_result, self.service.build_prompt(incident, rca_result, snapshot)
                )
                report.mode = "ml-sufficient"
                report.skipped_reason = (
                    f"Deterministic confidence {confidence:.2f} >= "
                    f"threshold {self.llm_confidence_threshold:.2f}; skipped LLM call."
                )
                report.usage["provider"] = "ml-sufficient"
                return report

        report = self.service.investigate(incident, rca_result, snapshot)
        if report.mode in {"ollama", "openai"}:
            report.mode = "llm"
        report.skipped_reason = None
        return report


__all__ = [
    "AgentUsage",
    "EvidenceItem",
    "HistoricalIncident",
    "IncidentInvestigator",
    "InvestigationService",
    "InvestigationToolbelt",
    "ToolTrace",
    "build_default_historical_incidents",
]