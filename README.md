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

Phase 4 adds:

- tool-based LLM investigation
- grounded evidence summaries
- historical incident memory
- request/token/cost tracking
- optional local Ollama-backed investigations

See [docs/architecture/phase-4.md](docs/architecture/phase-4.md) for the investigation layer.

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
See [docs/architecture/phase-4.md](docs/architecture/phase-4.md) for the evidence-grounded investigation layer.

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

To use the free local Ollama path instead of OpenAI API calls:

```powershell
ollama pull llama3.1
$env:MII_LLM_PROVIDER="ollama"
$env:MII_OLLAMA_MODEL="llama3.1"
$env:MII_OLLAMA_HOST="http://127.0.0.1:11434"
python -m mii.server
```

If Ollama is not running or the model is missing, the app falls back to the grounded local narrative.

Then open:

- `http://localhost:8080/`
- `http://localhost:8080/api/state`
- `http://localhost:8080/api/graph`
- `http://localhost:8080/api/rca`
- `http://localhost:8080/api/investigation`

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
  investigation.py  # tool-based LLM investigation
  state.py          # runtime orchestration
  dashboard.py      # HTML rendering
  server.py         # HTTP entry point
tests/
docs/architecture/
```
