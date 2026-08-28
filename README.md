# ML Incident Intelligence Platform

ML Incident Intelligence Platform is an end-to-end prototype for detecting and investigating production incidents in ML systems.

This repository starts with **Phase 1** only:

- synthetic telemetry generation
- streaming anomaly detection
- incident creation
- dashboard rendering

Phase 2 adds:

- feature drift detection
- data quality monitoring
- drift and quality panels in the dashboard

Phase 3 adds:

- dependency graph modeling
- graph-based root cause analysis
- ranked RCA hypotheses with evidence

The design intentionally leaves room for later phases such as tool-using LLM investigation, but those are not implemented yet.

## Problem

ML systems fail in ways that are harder to diagnose than ordinary application outages.

Signals are scattered across:

- model predictions
- feature availability
- latency
- error rates
- traffic patterns

Phase 1 focuses on creating a working vertical slice that can surface correlated telemetry changes as an incident.

## Architecture

```text
Synthetic Workload
    |
    v
Telemetry Generator
    |
    v
Rolling Anomaly Detectors
    |
    v
Incident Engine
    |
    v
Dashboard + JSON API
```

See [docs/architecture/phase-1.md](docs/architecture/phase-1.md) for the Phase 1 trade-offs and scope.
See [docs/architecture/phase-2.md](docs/architecture/phase-2.md) for the Phase 2 monitoring layer.
See [docs/architecture/phase-3.md](docs/architecture/phase-3.md) for the dependency graph and RCA layer.

## Phase 1 Scope

Implemented:

- deterministic synthetic telemetry
- fault injection
- rolling z-score detector
- EWMA-based detector
- incident aggregation
- live dashboard
- tests for the main slice

Not yet implemented:

- Kafka
- PostgreSQL
- LLM agent
- experimentation pipeline

## Quick Start

```bash
docker compose up --build
```

Or run locally:

```bash
python -m mii.server
```

Then open:

- `http://localhost:8080/`
- `http://localhost:8080/api/state`
- `http://localhost:8080/api/graph`
- `http://localhost:8080/api/rca`

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Repository Layout

```text
src/mii/
  synthetic.py      # telemetry + fault injection
  detectors.py      # streaming anomaly detectors
  incidents.py       # incident aggregation
  monitoring.py     # drift + data quality monitoring
  graph.py          # dependency graph + RCA
  state.py          # runtime orchestration
  dashboard.py      # HTML rendering
  server.py         # HTTP entry point
tests/
docs/architecture/
```
