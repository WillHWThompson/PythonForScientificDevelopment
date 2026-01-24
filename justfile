# Defaults
config := "configs/nlp_baseline.yaml"
profile := "workflow/profiles/local"

## Run the full pipeline (Train + Test)
all config=config profile=profile:
    uv run snakemake --workflow-profile {{profile}} --configfile {{config}}

## Run ONLY training for a specific config
train config=config profile=profile:
    uv run snakemake --workflow-profile {{profile}} --configfile {{config}} --allowed-rules train_bert_classifier

## Run ONLY evaluation for a specific config
test config=config profile=profile:
    uv run snakemake test_all --workflow-profile {{profile}} --configfile {{config}}

## Sync training results from VACC cluster
sync:
    ./tools/sync_vacc.sh

## Unlock snakemake directory if a process crashed
unlock:
    uv run snakemake --unlock

## Run local Quarto preview
preview:
    cd notebooks/quarto && quarto preview . --no-browser
