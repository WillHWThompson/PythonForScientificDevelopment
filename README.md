# PythonForScientificDevelopment

![Status](https://img.shields.io/badge/status-research%20artifact-blue)
![Language](https://img.shields.io/badge/language-Python-informational)
![Workflow](https://img.shields.io/badge/workflow-Snakemake-success)

A presentation-ready scientific software repository for NLP experiments with modular BERT adapters, reproducible workflow orchestration, and Quarto-based analysis outputs.

## Project Highlights
- Adapter-based DistilBERT training with a modular `src/` layout.
- Reproducible local and SLURM workflow execution via Snakemake profiles.
- Config-driven experiment design using YAML in `configs/`.
- Results aggregation and scientific reporting through Quarto docs and dashboards.

## Repository Layout
- `src/` — package code (`models`, `data`, `core`)
- `configs/` — baseline and sweep experiment definitions
- `scripts/` — entry points for training, testing, and plotting
- `workflow/` — Snakemake rules and execution profiles (`local`, `slurm`, `vacc`)
- `notebooks/` — Quarto notebooks and exploratory analysis
- `docs/` — generated rendered docs/site artifacts
- `tests/` — unit and integration tests

## Quickstart
```bash
cd PythonForScientificDevelopment
uv sync
```

Run a local baseline workflow:

```bash
uv run snakemake \
  --workflow-profile workflow/profiles/local \
  --configfile configs/nlp_baseline.yaml
```

## Example Notebook
Start with:
- `notebooks/quickstart_overview.ipynb`

Then explore:
- `notebooks/quarto/guide.qmd`
- `notebooks/quarto/dashboard.qmd`
- `notebooks/writeup.qmd`

## Cluster Notes (SLURM/VACC)
Use the `workflow/profiles/slurm` or `workflow/profiles/vacc` profile for cluster runs. Ensure the account/partition fields in profile config files match your cluster allocation before submitting large sweeps.

## Project Status
- Maturity: research-grade template / educational artifact
- Focus: software engineering patterns for computational science in Python
- Scope: optimized for clarity and reproducibility over production deployment hardening

## Author
Will Thompson
