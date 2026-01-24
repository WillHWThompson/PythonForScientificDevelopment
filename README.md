# BERT Text Classification with Modular Adapters

A professional scientific research template for NLP, demonstrating **Parameter-Efficient Fine-Tuning (PEFT)** using DistilBERT and Linear Adapters. This project uses the **AG News** dataset (120,000 headlines) to classify news into four categories: World, Sports, Business, and Sci/Tech.

## 🚀 Key Features

- **Functional Architecture**: Modular codebase organized into `models/`, `data/`, and `core/` for high maintainability.
- **Parameter Efficiency**: Uses a Linear Bottleneck Adapter (our MLP head) over a frozen DistilBERT backbone. Only ~1% of parameters are trainable.
- **Dynamic Hyperparameter Sweeps**: A robust Snakemake orchestration layer that automatically handles experiment grids.
- **Self-Validating Schemas**: Advanced Pydantic schemas with automatic experiment name generation and scientific parameter validation.
- **DuckDB Results Aggregation**: High-performance analysis using DuckDB to scan experiment JSONs directly into a research dashboard.

## 📁 Project Structure

```text
├── configs/            # YAML experiment configurations
├── scripts/            # Training entry points
├── src/
│   ├── models/         # BERT architectures and Adapter heads
│   ├── data/           # Hugging Face data loaders and tokenizers
│   └── core/           # Schemas, Trainers, and core logic
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

View your results aggregated via DuckDB in an interactive dashboard:

```bash
cd notebooks
quarto preview dashboard.qmd
```
