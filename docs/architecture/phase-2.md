# Phase 2 Architecture

Phase 2 adds monitoring for **feature drift** and **data quality** on top of the Phase 1 anomaly pipeline.

## Goal

Extend the synthetic environment so the platform can detect:

- population drift in feature distributions
- missing feature rates
- out-of-range feature values
- duplicate request telemetry

## Data Flow

```text
Synthetic Workload
    |
    +--> Serving Metrics --> Phase 1 Detectors --> Incident Engine
    |
    +--> Feature Observations --> Drift Monitor --> Drift Findings
    |
    +--> Feature Observations --> Data Quality Monitor --> Quality Issues
```

## Trade-offs

### Why PSI?

PSI is simple, interpretable, and easy to explain in a portfolio project. It is not the most statistically rigorous choice for every feature type, but it is a practical baseline that production teams often understand quickly.

### Why monitor missingness separately?

Drift and data quality are related but not identical. A feature can keep the same distribution shape while still losing a large fraction of values, so the platform should surface both signals.

### Why keep it synthetic?

The project still needs deterministic tests and controlled fault injection before moving to more complex data sources and infrastructure.

## Exit Criteria

Phase 2 is complete when:

- drift findings appear after the synthetic feature shift
- quality issues appear after the synthetic missingness fault
- the dashboard surfaces both signals
- the test suite covers the new monitors

