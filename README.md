# Scientific Python Development: Higgs Boson Classification
# BERT Text Classification with Adapters

A professional scientific research template for NLP, demonstrating **Parameter-Efficient Fine-Tuning (PEFT)** using DistilBERT and Linear Adapters. This project uses the **AG News** dataset (120,000 headlines) to classify news into four categories: World, Sports, Business, and Sci/Tech.

## 🚀 Key Features

- **Parameter Efficiency**: Uses a Linear Bottleneck Adapter head over a frozen DistilBERT backbone. Only ~1% of parameters are trainable.
- **Dynamic Hyperparameter Sweeps**: A robust Snakemake orchestration layer that automatically handles experiment grids (learning rate, adapter dimension, etc.).
- **Professional NLP Stack**: Built with PyTorch, Hugging Face `transformers`, `datasets`, and `evaluate`.
- **DuckDB Results Aggregation**: High-performance results analysis using DuckDB to scan experiment JSONs directly into a research dashboard.
- **Quarto Dashboards**: Interactive research website to visualize experiment metrics and model performance.

## 📁 Project Structure

```text
├── configs/            # YAML experiment configurations
├── scripts/            # Training entry points
├── src/
│   └── text_classifier/ # Core model, data, and trainer logic
├── workflow/
│   ├── rules/          # Modular Snakemake rules
│   └── profiles/       # Site-specific executor settings (Slurm, Local)
├── notebooks/          # Quarto analysis and dashboard
└── Snakefile           # Main workflow orchestrator
```

## 🛠 Setup & Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Install dependencies
uv sync
```

## 📈 Running Experiments

We use **Snakemake Profiles** found in `workflow/profiles/` to manage execution across different environments.

### 1. Basic Experiment
```bash
uv run snakemake --workflow-profile workflow/profiles/local --configfile configs/nlp_baseline.yaml
```

### 2. High-Performance Sweeps (Slurm)
The pipeline automatically detects the `sweep` section in your YAML and parallelizes the jobs.
```bash
uv run snakemake --workflow-profile workflow/profiles/slurm --configfile configs/nlp_baseline.yaml
```

## 📊 Research Dashboard

View your results in an interactive dashboard:

```bash
cd notebooks
quarto preview dashboard.qmd
```

The dashboard uses DuckDB to aggregate stats from all experiment subfolders in `results/`, providing a unified view of your research progress.

## 🧪 Testing and CI

Run unit tests locally:
```bash
uv run pytest tests/
```

This project uses **GitHub Actions** to automatically run linting (`ruff`) and tests on every pull request.

## 📂 Project Structure

```text
.
├── .github/workflows/   # CI/CD (GitHub Actions)
├── data/                # Data storage (Large files excluded from Git)
├── notebooks/           # Quarto research website (index, daily-notes, writeup, dashboard)
├── scripts/             # Training and experiment entry points
├── src/scientific_dev/  # Core logic (JAX models, DuckDB managers, Pydantic schemas)
├── tests/               # Unit tests for core components
├── pyproject.toml       # Optimized dependency definitions with uv
└── Snakefile            # Snakemake workflow orchestration
```
