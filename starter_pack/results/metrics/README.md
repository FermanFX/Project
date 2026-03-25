# `metrics/`

Per-run metrics and performance scores.

## Purpose

Store individual run outputs indexed by run ID or configuration.

## Organization

Organize by run ID or date:
```
metrics/
├── run_20260325_001/
│   └── metrics.json
├── run_20260325_002/
│   └── metrics.json
└── ...
```

## Content

Include metrics such as:
- Loss/accuracy over training
- Validation scores
- Final test performance
- Inference time
- Memory usage

Metadata:
- Seed value
- Hyperparameters
- Timestamp
- Git commit (if applicable)
