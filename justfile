# Defaults
config := "configs/nlp_baseline.yaml"
local_profile := "workflow/profiles/local"
slurm_profile := "workflow/profiles/slurm"

## Run the full pipeline (Train + Test)
all config=config:
    uv run snakemake --workflow-profile {{local_profile}} --configfile {{config}}

## [HPC] Run the full pipeline on cluster
all-hpc config=config:
    uv run snakemake --workflow-profile {{slurm_profile}} --configfile {{config}}

## Run ONLY training locally
train config=config:
    uv run snakemake --workflow-profile {{local_profile}} --configfile {{config}} --allowed-rules train_bert_classifier

## [HPC] Run ONLY training on cluster
train-hpc config=config:
    uv run snakemake --workflow-profile {{slurm_profile}} --configfile {{config}} --allowed-rules train_bert_classifier

## Run ONLY evaluation locally
test config=config:
    uv run snakemake test_all --workflow-profile {{local_profile}} --configfile {{config}}

## [HPC] Run ONLY evaluation on cluster
test-hpc config=config:
    uv run snakemake test_all --workflow-profile {{slurm_profile}} --configfile {{config}}

## Sync training results from VACC cluster
sync:
    ./tools/sync_vacc.sh

## Unlock snakemake directory if a process crashed
unlock:
    uv run snakemake --unlock

## Run local Quarto preview
preview:
    cd notebooks/quarto && quarto preview . --no-browser
