# `results/`

Reproducible experiment outputs organized by type.

## Structure

- **`tables/`** – Summary and comparison tables (CSV, JSON)
- **`metrics/`** – Per-run metrics and performance scores
- **`statistics/`** – Aggregated statistics across multiple seeds/runs
- **`logs/`** – Experiment logs and traces

## Guidelines

- Keep outputs lightweight and reproducible
- Include metadata (seed, hyperparameters, timestamp)
- Document the source or generation script
- Avoid storing raw training data or large artifacts
- Use structured formats (CSV, JSON) for easy parsing
