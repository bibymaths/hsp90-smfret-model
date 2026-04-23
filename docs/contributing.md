# Contributing

## Development environment

```bash
git clone <repo-url>
cd hsp90-smfret-model
pip install -e ".[dev,docs]"
```

## Quality checks

```bash
pytest --cov
ruff check .
ruff format .
```

## Branch and PR conventions

- Branch naming: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `ci/<topic>`.
- Keep commits focused and descriptive.
- Open PRs with:
  - summary of scientific/technical intent,
  - validation notes (tests + lint),
  - output artifacts impacted (tables/plots/docs).

## CI workflows

- **Lint workflow** (`.github/workflows/lint.yml`) enforces Ruff formatting and linting.
- **Tests workflow** (`.github/workflows/tests.yml`) runs pytest with coverage gates and uploads coverage.
