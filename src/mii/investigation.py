from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from typing import Any, TYPE_CHECKING

from .graph import DeploymentEvent, RootCauseHypothesis
from .monitoring import DataQualityIssue, DriftFinding, FeatureObservation

if TYPE_CHECKING:
    from .state import PhaseOneSnapshot


@dataclass(slots=True)
class HistoricalIncident:
    """A past incident used to ground the investigation in memory."""

    incident_id: str
    title: str
    component: str
    root_cause: str
    summary: str
    lessons: list[str]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "component": self.component,
            "root_cause": self.root_cause,
            "summary": self.summary,
            "lessons": list(self.lessons),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_report(cls, report: "InvestigationReport") -> "HistoricalIncident":
        return cls(
            incident_id=report.incident_id,
            title=f"Resolved investigation for {report.verdict}",
            component=report.verdict_component,
            root_cause=report.verdict,
            summary=report.summary,
            lessons=list(report.recommended_actions[:3]),
            timestamp=report.generated_at,
        )


@dataclass(slots=True)
class EvidenceItem:
    """A grounded evidence item attached to the investigation report."""

    source: str
    label: str
    detail: str
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "detail": self.detail,
            "citation": self.citation,
        }


@dataclass(slots=True)
class ToolTrace:
    """Trace of a local tool used by the agent."""

    tool_name: str
    purpose: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "purpose": self.purpose,
            "summary": self.summary,
        }


@dataclass(slots=True)
class AgentUsage:
    """Estimated or observed usage for the investigation agent."""

    provider: str
    model: str
    mode: str
    requests: int
    input_tokens: int
    output_tokens: int
    estimated_latency_ms: float
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(slots=True)
class InvestigationReport:
    """Grounded incident investigation produced by the Phase 4 agent."""

    incident_id: str
    generated_at: datetime
    mode: str
    verdict: str
    verdict_component: str
    confidence: float
    summary: str
    recommended_actions: list[str]
    evidence: list[EvidenceItem]
    historical_matches: list[HistoricalIncident]
    tool_trace: list[ToolTrace]
    usage: AgentUsage
    llm_note: str | None = None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "generated_at": self.generated_at.isoformat(),
            "mode": self.mode,
            "verdict": self.verdict,
            "verdict_component": self.verdict_component,
            "confidence": self.confidence,
            "summary": self.summary,
            "recommended_actions": list(self.recommended_actions),
            "evidence": [item.to_dict() for item in self.evidence],
            "historical_matches": [item.to_dict() for item in self.historical_matches],
            "tool_trace": [trace.to_dict() for trace in self.tool_trace],
            "usage": self.usage.to_dict(),
            "llm_note": self.llm_note,
            "skipped_reason": self.skipped_reason,
        }


def build_default_historical_incidents() -> list[HistoricalIncident]:
    """Seed the agent with a few realistic prior incidents."""

    return [
        HistoricalIncident(
            incident_id="INC-1204",
            title="Feature normalization rollout caused missing inputs",
            component="feature-service",
            root_cause="Feature-service deployment",
            summary="A feature-service rollout changed normalization logic and increased missing input rates.",
            lessons=[
                "Correlate missingness spikes with deployment timelines.",
                "Validate upstream feature contracts after rollouts.",
                "Rollback should be the first mitigation for feature-service regressions.",
            ],
            timestamp=datetime(2026, 6, 14, 10, 10, tzinfo=timezone.utc),
        ),
        HistoricalIncident(
            incident_id="INC-1268",
            title="Data pipeline lag caused stale training features",
            component="data-pipeline",
            root_cause="Delayed batch ingestion",
            summary="A delayed batch job propagated stale features into the serving path and increased drift.",
            lessons=[
                "Monitor ingestion lag and freshness separately from serving latency.",
                "RCA should consider upstream data freshness as a root cause.",
            ],
            timestamp=datetime(2026, 7, 3, 8, 45, tzinfo=timezone.utc),
        ),
        HistoricalIncident(
            incident_id="INC-1331",
            title="Model serving latency increased after gateway pressure",
            component="api-gateway",
            root_cause="Gateway saturation",
            summary="A gateway capacity issue increased request latency and amplified error rates.",
            lessons=[
                "Differentiate downstream latency symptoms from upstream root causes.",
                "Use dependency graphs to avoid misattributing the blast radius.",
            ],
            timestamp=datetime(2026, 7, 28, 14, 5, tzinfo=timezone.utc),
        ),
    ]


class InvestigationToolbelt:
    """Local tools used by the investigation agent."""

    def __init__(
        self,
        *,
        snapshot: PhaseOneSnapshot,
        historical_incidents: list[HistoricalIncident],
    ) -> None:
        self.snapshot = snapshot
        self.historical_incidents = historical_incidents

    def incident_context(self) -> dict[str, Any]:
        incident = self.snapshot.incident
        if incident is None:
            return {"status": "no incident"}

        top_hypothesis = self.top_hypothesis()
        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity,
            "confidence": incident.confidence,
            "summary": incident.summary,
            "top_hypothesis": None if top_hypothesis is None else top_hypothesis.to_dict(),
        }

    def dependency_graph(self) -> dict[str, Any]:
        return self.snapshot.dependency_graph.to_dict()

    def evidence_bundle(self) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for anomaly in self.snapshot.recent_anomalies[-8:]:
            evidence.append(
                EvidenceItem(
                    source="anomaly",
                    label=anomaly.metric,
                    detail=f"{anomaly.explanation} (score={anomaly.z_score:.2f})",
                    citation=f"anomaly:{anomaly.metric}:{anomaly.timestamp.isoformat()}",
                )
            )
        for finding in self.snapshot.drift_findings[-6:]:
            evidence.append(
                EvidenceItem(
                    source="drift",
                    label=finding.feature_name,
                    detail=f"PSI={finding.psi:.3f}; {finding.explanation}",
                    citation=f"drift:{finding.feature_name}:{finding.timestamp.isoformat()}",
                )
            )
        for issue in self.snapshot.quality_issues[-6:]:
            evidence.append(
                EvidenceItem(
                    source="quality",
                    label=issue.feature_name,
                    detail=f"{issue.issue_type}: {issue.description}",
                    citation=f"quality:{issue.feature_name}:{issue.timestamp.isoformat()}",
                )
            )
        for deployment in self.snapshot.deployment_events[-4:]:
            evidence.append(
                EvidenceItem(
                    source="deployment",
                    label=deployment.component,
                    detail=f"{deployment.version} at step {deployment.step}: {deployment.description}",
                    citation=f"deployment:{deployment.component}:{deployment.timestamp.isoformat()}",
                )
            )
        return evidence

    def top_hypothesis(self) -> RootCauseHypothesis | None:
        return self.snapshot.root_cause_hypotheses[0] if self.snapshot.root_cause_hypotheses else None

    def historical_matches(self) -> list[HistoricalIncident]:
        top = self.top_hypothesis()
        if top is None:
            return self.historical_incidents[:2]

        label_head = top.label.lower().split()[0] if top.label else top.component
        keywords = {top.component, top.label.lower(), label_head}
        matches = [
            incident
            for incident in self.historical_incidents
            if incident.component in keywords or incident.root_cause.lower().startswith(top.label.lower().split()[0])
        ]
        return matches[:3] or self.historical_incidents[:2]

    def cost_profile(self) -> dict[str, float]:
        evidence_json = json.dumps([item.to_dict() for item in self.evidence_bundle()], ensure_ascii=True)
        incident_json = json.dumps(self.incident_context(), ensure_ascii=True)
        prompt_text = evidence_json + incident_json
        input_tokens = _estimate_tokens(prompt_text)
        output_tokens = 240
        input_rate = float(os.getenv("MII_LLM_INPUT_COST_PER_1M", "4.0"))
        output_rate = float(os.getenv("MII_LLM_OUTPUT_COST_PER_1M", "20.0"))
        estimated_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        }


class IncidentInvestigator:
    """Tool-based incident investigator with optional local or OpenAI text generation."""

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        llm_provider: str = "ollama",
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        llm_confidence_threshold: float = 0.96,
        max_output_tokens: int = 300,
    ) -> None:
        self.model = model
        self.llm_provider = llm_provider
        self.ollama_host = (ollama_host or os.getenv("MII_OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = ollama_model or os.getenv("MII_OLLAMA_MODEL", self.model)
        self.llm_confidence_threshold = llm_confidence_threshold
        self.max_output_tokens = max_output_tokens

    def investigate(
        self,
        *,
        snapshot: PhaseOneSnapshot,
        historical_incidents: list[HistoricalIncident],
    ) -> InvestigationReport:
        if snapshot.incident is None:
            usage = AgentUsage(
                provider="local",
                model=self.model,
                mode="idle",
                requests=0,
                input_tokens=0,
                output_tokens=0,
                estimated_latency_ms=0.0,
                estimated_cost_usd=0.0,
            )
            return InvestigationReport(
                incident_id="none",
                generated_at=datetime.now(timezone.utc),
                mode="idle",
                verdict="no active incident",
                verdict_component="none",
                confidence=0.0,
                summary="No incident is active, so the investigator is idle.",
                recommended_actions=[],
                evidence=[],
                historical_matches=[],
                tool_trace=[],
                usage=usage,
            )

        toolbelt = InvestigationToolbelt(snapshot=snapshot, historical_incidents=historical_incidents)
        tool_trace = [
            ToolTrace("incident_context", "capture the active incident and the leading hypothesis", "Active incident context assembled."),
            ToolTrace("dependency_graph", "inspect blast radius and upstream dependencies", "Dependency graph loaded."),
            ToolTrace("evidence_bundle", "collect grounded evidence", f"{len(toolbelt.evidence_bundle())} evidence items collected."),
            ToolTrace("historical_matches", "retrieve similar incidents", f"{len(toolbelt.historical_matches())} historical matches found."),
            ToolTrace("cost_profile", "budget the investigation", "Token and cost estimate computed."),
        ]

        top = toolbelt.top_hypothesis()
        evidence = toolbelt.evidence_bundle()
        historical_matches = toolbelt.historical_matches()
        should_call_llm = self._should_call_llm(snapshot, top)
        usage_profile = toolbelt.cost_profile()

        mode = "tool-grounded"
        resolved_provider = "local"
        llm_note: str | None = None
        skipped_reason: str | None = None

        provider = self.llm_provider.lower().strip()

        if should_call_llm and provider == "openai":
            llm_note, usage_profile, skipped_reason = self._call_openai(toolbelt, usage_profile)
            resolved_provider = "openai"
            mode = "llm" if skipped_reason is None else "local-fallback"
            if skipped_reason is not None:
                llm_note = self._local_narrative(toolbelt, historical_matches)
        elif should_call_llm and provider == "ollama":
            llm_note, usage_profile, skipped_reason = self._call_ollama(toolbelt, usage_profile)
            resolved_provider = "ollama"
            mode = "llm" if skipped_reason is None else "local-fallback"
            if skipped_reason is not None:
                llm_note = self._local_narrative(toolbelt, historical_matches)
        elif should_call_llm and provider == "auto":
            llm_note, usage_profile, skipped_reason = self._call_ollama(toolbelt, usage_profile)
            if skipped_reason is None:
                resolved_provider = "ollama"
                mode = "llm"
            else:
                llm_note, usage_profile, skipped_reason = self._call_openai(toolbelt, usage_profile)
                resolved_provider = "openai"
                mode = "llm" if skipped_reason is None else "local-fallback"
            if skipped_reason is not None:
                llm_note = self._local_narrative(toolbelt, historical_matches)
        elif should_call_llm:
            mode = "local-fallback"
            skipped_reason = f"unsupported llm provider: {self.llm_provider}"
            llm_note = self._local_narrative(toolbelt, historical_matches)
        else:
            mode = "ml-sufficient"
            skipped_reason = "ML/RCA confidence was high enough to skip the LLM call"
            llm_note = self._local_narrative(toolbelt, historical_matches)

        confidence = round(min(0.99, max(snapshot.incident.confidence, top.confidence if top else 0.0)), 2)
        verdict_component = top.component if top is not None else "unknown"
        verdict = top.label if top is not None else snapshot.incident.title
        summary = llm_note or self._local_narrative(toolbelt, historical_matches)
        recommended_actions = self._recommend_actions(snapshot, top, historical_matches)
        usage = AgentUsage(
            provider=resolved_provider if mode == "llm" else "local",
            model=self.ollama_model if mode == "llm" and resolved_provider == "ollama" else self.model,
            mode=mode,
            requests=1 if mode == "llm" else 0,
            input_tokens=usage_profile["input_tokens"],
            output_tokens=usage_profile["output_tokens"],
            estimated_latency_ms=usage_profile.get("estimated_latency_ms", 0.0),
            estimated_cost_usd=usage_profile["estimated_cost_usd"],
        )

        return InvestigationReport(
            incident_id=snapshot.incident.incident_id,
            generated_at=datetime.now(timezone.utc),
            mode=mode,
            verdict=verdict,
            verdict_component=verdict_component,
            confidence=confidence,
            summary=summary,
            recommended_actions=recommended_actions,
            evidence=evidence,
            historical_matches=historical_matches,
            tool_trace=tool_trace,
            usage=usage,
            llm_note=llm_note if mode == "llm" else None,
            skipped_reason=skipped_reason,
        )

    def _should_call_llm(self, snapshot: PhaseOneSnapshot, top: RootCauseHypothesis | None) -> bool:
        if snapshot.incident is None:
            return False
        if top is None:
            return True
        if snapshot.incident.confidence < self.llm_confidence_threshold:
            return True
        if top.confidence < self.llm_confidence_threshold:
            return True
        if len(snapshot.root_cause_hypotheses) > 1 and snapshot.root_cause_hypotheses[0].score - snapshot.root_cause_hypotheses[1].score < 0.12:
            return True
        return False

    def _call_openai(
        self,
        toolbelt: InvestigationToolbelt,
        usage_profile: dict[str, float],
    ) -> tuple[str, dict[str, float], str | None]:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, str(exc)

        try:
            client = OpenAI()
        except Exception as exc:  # pragma: no cover - optional dependency/runtime issue
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"openai client init failed: {exc}"

        prompt = self._build_prompt(toolbelt)
        start = time.perf_counter()
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You are a grounded incident investigator for an ML platform. "
                                    "Use only the provided evidence and history. Do not invent facts. "
                                    "Return a concise, useful investigation summary."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/auth/runtime issue
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"openai request failed: {exc}"

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output_text = getattr(response, "output_text", "") or self._local_narrative(toolbelt, toolbelt.historical_matches())
        usage_profile["estimated_latency_ms"] = round(elapsed_ms, 1)
        usage_profile["output_tokens"] = max(usage_profile.get("output_tokens", 0), _estimate_tokens(output_text))
        usage_profile["estimated_cost_usd"] = _estimate_cost(usage_profile["input_tokens"], usage_profile["output_tokens"])
        return output_text.strip(), usage_profile, None

    def _call_ollama(
        self,
        toolbelt: InvestigationToolbelt,
        usage_profile: dict[str, float],
    ) -> tuple[str, dict[str, float], str | None]:
        prompt = self._build_prompt(toolbelt)
        payload = {
            "model": self.ollama_model,
            "prompt": (
                "You are a grounded incident investigator for an ML platform. "
                "Use only the provided evidence and history. Do not invent facts. "
                "Return a concise, useful investigation summary.\n\n"
                f"{prompt}"
            ),
            "stream": False,
            "options": {
                "num_predict": self.max_output_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.ollama_host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - network/runtime issue
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"ollama request failed: {exc}"
        except URLError as exc:  # pragma: no cover - network/runtime issue
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"ollama request failed: {exc}"
        except Exception as exc:  # pragma: no cover - network/runtime issue
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"ollama request failed: {exc}"

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            decoded = json.loads(raw)
            output_text = str(decoded.get("response", "")).strip()
        except Exception as exc:  # pragma: no cover - malformed response
            return self._local_narrative(toolbelt, toolbelt.historical_matches()), usage_profile, f"ollama response parse failed: {exc}"

        if not output_text:
            output_text = self._local_narrative(toolbelt, toolbelt.historical_matches())

        usage_profile["estimated_latency_ms"] = round(elapsed_ms, 1)
        usage_profile["output_tokens"] = max(usage_profile.get("output_tokens", 0), _estimate_tokens(output_text))
        usage_profile["estimated_cost_usd"] = 0.0
        return output_text, usage_profile, None

    def _build_prompt(self, toolbelt: InvestigationToolbelt) -> str:
        payload = {
            "incident_context": toolbelt.incident_context(),
            "evidence": [item.to_dict() for item in toolbelt.evidence_bundle()],
            "historical_matches": [item.to_dict() for item in toolbelt.historical_matches()],
            "dependency_graph": toolbelt.dependency_graph(),
            "cost_profile": toolbelt.cost_profile(),
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _local_narrative(
        self,
        toolbelt: InvestigationToolbelt,
        historical_matches: list[HistoricalIncident],
    ) -> str:
        top = toolbelt.top_hypothesis()
        if top is None:
            return "No grounded root cause could be established."

        evidence_lines = [
            f"- {item.label}: {item.detail}"
            for item in toolbelt.evidence_bundle()[:4]
        ]
        history_lines = [
            f"- {incident.incident_id}: {incident.title} -> {incident.root_cause}"
            for incident in historical_matches[:3]
        ]
        return (
            f"Grounded investigation points to {top.label} ({top.component}) with confidence {top.confidence:.2f}. "
            f"Evidence: {'; '.join(evidence_lines) if evidence_lines else 'none'}. "
            f"Historical parallels: {'; '.join(history_lines) if history_lines else 'none'}."
        )

    def _recommend_actions(
        self,
        snapshot: PhaseOneSnapshot,
        top: RootCauseHypothesis | None,
        historical_matches: list[HistoricalIncident],
    ) -> list[str]:
        actions = list(snapshot.incident.recommended_actions)
        if top is not None:
            actions.extend(
                [
                    f"Prioritize remediation for {top.label.lower()} ({top.component}).",
                    "Ground the rollback decision in the evidence bundle before broad mitigation.",
                ]
            )
        if historical_matches:
            actions.append(f"Review historical incidents: {', '.join(item.incident_id for item in historical_matches[:2])}.")
        return _dedupe_preserve_order(actions)


def _openai_available() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    return importlib.util.find_spec("openai") is not None


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_rate = float(os.getenv("MII_LLM_INPUT_COST_PER_1M", "4.0"))
    output_rate = float(os.getenv("MII_LLM_OUTPUT_COST_PER_1M", "20.0"))
    return round((input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate, 6)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
