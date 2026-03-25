# `statistics/`

Aggregated statistics across multiple seeds and runs.

## Purpose

Store repeated-seed statistics and aggregate analysis across runs.

## Content

- Mean and standard deviation across seeds
- Confidence intervals
- Statistical test results
- Distribution summaries

## Format

JSON or CSV with:
- `mean` – Aggregate metric value
- `std` – Standard deviation
- `min` / `max` – Range
- `seed_count` – Number of seeds used
- Individual seed raw values (optional)

## Examples

```json
{
  "model": "baseline",
  "accuracy_mean": 0.92,
  "accuracy_std": 0.015,
  "accuracy_seeds": [0.918, 0.925, 0.919],
  "seeds": 3
}
```
