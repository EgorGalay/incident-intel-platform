# Phase 4 Architecture

Phase 4 adds a tool-based LLM investigator on top of the graph RCA layer.

## Goal

Transform the platform from:

- "we can detect and rank likely causes"

into:

- "we can explain an incident using grounded evidence, prior incidents, and a controlled LLM investigation"

## Investigation Flow

```text
Active incident
    |
    v
Local tools gather evidence
    |
    +--> anomaly feed
    +--> drift findings
    +--> data quality issues
    +--> deployment events
    +--> dependency graph
    +--> historical incidents
    |
    v
Cost-aware gate
    |
    +--> skip LLM if ML/RCA is already sufficient
    +--> otherwise call OpenAI Responses API when available
    |
    v
Grounded investigation report
```

## Tooling

The agent relies on local tools for:

- incident context
- evidence bundle generation
- dependency graph inspection
- historical incident matching
- cost estimation

The LLM is used to turn grounded tool output into a concise investigation narrative, not to invent the incident from scratch.

## Trade-offs

### Why keep the tools local?

It keeps the agent deterministic, testable, and useful even when the API is unavailable.

### Why allow skipping the LLM?

Production systems should not burn tokens when the ML and RCA stack already has enough confidence to explain the incident.

### Why seed historical incidents?

The agent needs memory to compare the current incident against prior patterns before moving to a live incident store.

## Exit Criteria

Phase 4 is complete when:

- the dashboard shows an investigation report
- historical incidents are available to the agent
- evidence is explicitly grounded in telemetry and graph signals
- the system tracks requests, tokens, and estimated cost
- the agent can skip the LLM when the ML/RCA layer is sufficient

