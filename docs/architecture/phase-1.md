# Phase 1 Architecture

Phase 1 is a narrow but complete vertical slice.

## Goal

Prove that the platform can:

- generate synthetic ML telemetry
- inject a realistic degradation
- detect anomalous behavior in a stream
- create an incident from correlated anomalies
- render the live state in a dashboard

## Scope

### In scope

- one synthetic workload
- one incident class: feature/service degradation
- streaming detectors
- incident aggregation
- JSON API
- HTML dashboard

### Out of scope

- Kafka
- PostgreSQL
- graph RCA
- LLM reasoning
- historical benchmarking
- distributed deployment

## Data Flow

```text
Synthetic Workload
    |
    v
Telemetry Samples
    |
    v
Rolling Z-Score Detector
    |
    v
EWMA Detector
    |
    v
Incident Engine
    |
    v
Dashboard Snapshot
```

## Trade-offs

### Why start with deterministic synthetic telemetry?

Deterministic telemetry makes the system testable and debuggable before introducing distributed infrastructure.

### Why use streaming statistical detectors first?

They are easy to validate, cheap to run, and provide a strong baseline for later ML models.

### Why keep the dashboard simple?

The first milestone should prove signal flow and incident creation, not frontend complexity.

## Phase 1 Exit Criteria

Phase 1 is complete when:

- telemetry changes after fault injection
- detectors emit anomalies
- the incident engine creates a live incident
- the dashboard shows the incident and the anomaly summary
- tests cover the core behavior

