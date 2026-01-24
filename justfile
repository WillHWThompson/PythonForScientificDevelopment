# Default to baseline if no config is provided
config := "configs/nlp_baseline.yaml"

## Run the full pipeline (Train + Test)
all config=config:
    uv run snakemake --workflow-profile workflow/profiles/local --configfile {{config}}

## Run ONLY training for a specific config
train config=config:
    uv run snakemake --workflow-profile workflow/profiles/local --configfile {{config}} --allowed-rules train_bert_classifier

## Run ONLY evaluation for a specific config
test config=config:
    uv run snakemake --workflow-profile workflow/profiles/local --configfile {{config}} --allowed-rules test_bert_classifier

## Sync training results from VACC cluster
sync:
    ./tools/sync_vacc.sh

## Run local Quarto preview
preview:
    cd notebooks/quarto && quarto preview . --no-browser
