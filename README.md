# Scientific Python Development: Higgs Boson Classification

This repository demonstrates a professional, high-performance scientific development workflow in Python. It uses the **Higgs Boson Dataset** (11 Million samples) to showcase modern tools for orchestration, hardware acceleration, and data management.

## 🛠️ Technology Stack

- **[uv](https://github.com/astral-sh/uv)**: Extremely fast Python package and project manager.
- **[JAX](https://github.com/google/jax)**: Composable transformations of Python+NumPy (Autograd, XLA) for hardware-accelerated deep learning.
- **[Snakemake](https://snakemake.github.io/)**: Workflow management for reproducible and scalable data pipelines.
- **[DuckDB](https://duckdb.org/)**: High-performance analytical database for streaming large Parquet files.
- **[Weights & Biases (WandB)](https://wandb.ai/)**: Real-time experiment tracking and training visualization.
- **[Quarto](https://quarto.org/)**: Scientific and technical publishing system for research websites and interactive dashboards.
- **[Pydantic](https://docs.pydantic.dev/)**: Robust configuration and data validation.

## 🚀 Getting Started

### 1. Prerequisites
Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup the Environment
```bash
uv sync
```

### 3. Run the Pipeline
The workflow is orchestrated via Snakemake. It handles data download, conversion to Parquet, and model training.

**Dry Run:**
```bash
uv run snakemake -n
```

**Local Execution:**
```bash
uv run snakemake --executor local --cores 4
```

**HPC (Slurm) Execution:**
```bash
uv run snakemake --executor slurm --jobs 10
```

## 📊 Research Website & Dashboard

We use Quarto to render a cohesive research website from our notebooks.

```bash
cd notebooks
quarto preview
```

The website includes:
- **Daily Research Notes**: A chronological record of development.
- **Scientific Write-up**: A formal report with BibTeX citations.
- **Results Dashboard**: Interactive visualization of model performance and feature analysis.

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
