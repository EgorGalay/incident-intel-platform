# Phase 3 Architecture

Phase 3 introduces a dependency graph and graph-based root cause analysis.

## Goal

Move beyond anomaly detection and drift monitoring by answering:

- which component is most likely responsible for the incident?
- what evidence supports that hypothesis?
- what is the downstream blast radius?

## Graph Model

The dependency graph is intentionally small and explicit for Phase 3:

```text
data-pipeline -> feature-service -> model-serving -> api-gateway
                      \----------------------------^
```

The graph is static in this phase so that RCA behavior is deterministic and testable.

## RCA Flow

```text
Telemetry, drift, quality, deployment events
    |
    v
Signal aggregation by component
    |
    v
Graph propagation and evidence scoring
    |
    v
Ranked root cause hypotheses
```

## Trade-offs

### Why a static graph first?

It makes the causal reasoning explainable and keeps the experiment controlled. Later phases can replace the graph source with service discovery or metadata from production systems.

### Why score with both graph topology and evidence?

Topology alone is too weak, and evidence alone ignores blast radius. Combining them produces a better production-style RCA signal.

### Why return ranked hypotheses?

Incidents often have uncertainty. A ranked list is more useful than a single brittle answer and mirrors how SRE and ML platform teams investigate outages.

## Exit Criteria

Phase 3 is complete when:

- the dashboard shows the dependency graph
- RCA returns ranked hypotheses
- hypotheses include evidence and downstream impact
- tests validate the ranking behavior

