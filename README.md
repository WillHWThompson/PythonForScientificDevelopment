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

### 3. Setup Experiment Tracking (WandB)
To visualize your training live:
1.  **Login to WandB**:
    ```bash
    uv run wandb login
    ```
2.  **Update Config**: Open `config.yaml` and set `enabled: true` and your `entity` (username).

### 4. Run the Pipeline
The workflow is orchestrated via **Snakemake**. To run an experiment, you point to a specific configuration file using the `--configfile` flag.

**Example: Run a Sweep**
```bash
uv run snakemake --workflow-profile workflow/profiles/local --configfile configs/default.yaml
```

**Example: Run a Narrow Search**
```bash
uv run snakemake --workflow-profile workflow/profiles/local --configfile configs/narrow_sweep.yaml
```

## 🚀 Key Features
Profiles are the professional way to handle cluster portability:
- **Separation of Concerns**: Your `Snakefile` defines the science; your `Profile` defines the hardware.
- **No Batch Scripts**: Snakemake handles the `sbatch` submission for you.
- **Default Resources**: You can set global timeouts and memory limits in the profile, which rules can then override.

## 🔄 Modular Workflows

As projects grow, `Snakefiles` can become cluttered. We use **Snakemake Rules Modules** to keep things organized:
- `workflow/rules/data.smk`: Handles ingestion and preprocessing.
- `workflow/rules/train.smk`: Handles JAX model training.

The main `Snakefile` simply pulls these together using `include:`. This pattern allows you to share rules across different projects or workflows easily.

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
