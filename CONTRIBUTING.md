# Contributing to fedchem

Thanks for your interest in contributing! This short guide covers local setup, quality checks, and our CI gates.

## Development setup
- Requires Python 3.10–3.12
- Install dev deps: `pip install -e .[dev]`

## Pre‑commit hooks
- Install pre‑commit: `pip install pre-commit`
- Enable hooks: `pre-commit install`
- Run on all files once: `pre-commit run --all-files`

Hooks run Ruff, Black, mypy, basic hygiene checks, and a secrets scan.

## Lint & types
- Ruff (lint): `ruff check src`
- Mypy (types): `mypy src tests`

## Tests & coverage
- Run tests: `pytest -q`
- Coverage thresholds are enforced via `pyproject.toml` (lines >= 85%, branches >= 75%).

## Spec‑lint & governance
- Spec lint: `spec-lint` (or `python -m fedchem.scripts.spec_lint`)
- Traceability, A‑files (A1..A5), and DECISIONS.md should be kept current.

## CI
- GitHub Actions runs on Ubuntu and Windows for Python 3.10–3.12.
- Gates: Ruff, mypy, pytest + coverage, spec‑lint, SBOM (CycloneDX), license allowlist, secrets scan.

## Pull request checklist
- [ ] Code formatted and lint‑clean (Ruff/Black)
- [ ] Types clean (mypy)
- [ ] Tests added/updated; `pytest -q` passes locally
- [ ] Coverage not reduced (target ≥ 85% lines)
- [ ] Spec‑lint OK; A‑files, TRACEABILITY.csv, DECISIONS.md updated if applicable
- [ ] No secrets or license issues introduced

## Commit messages
- Use concise, descriptive messages (e.g., `feat:`, `fix:`, `chore:`). Keep related changes in a single commit when practical.

## Reporting issues
- Include environment (OS, Python), steps to reproduce, expected vs actual behavior, and logs where relevant.

