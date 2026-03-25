# `logs/`

Lightweight experiment logs and execution traces.

## Purpose

Store minimal logs for reproducibility and debugging. Focus on essential information only.

## Content

- Experiment parameters and configuration
- Training progress snapshots (e.g., every epoch)
- Error traces and warnings
- System info (versions, environment)

## Format

- Text logs (.txt, .log)
- JSON logs for structured parsing
- Keep file sizes under 10MB per log

## Examples

```
logs/
├── exp_20260325_001.log
├── exp_20260325_002.log
└── config_dump.json
```

## Guidelines

- Use timestamps for each entry
- Include seed and run ID in log filename
- Archive old logs if they accumulate
