# Default Paths
config := "configs/nlp_baseline.yaml"
local := "workflow/profiles/local"
slurm := "workflow/profiles/slurm"

## Run full pipeline. Usage: just all [config] [profile]
all c=config p=local:
    uv run snakemake --workflow-profile {{p}} --configfile {{c}}

## Run training. Usage: just train [config] [profile]
train c=config p=local:
    uv run snakemake --workflow-profile {{p}} --configfile {{c}} --allowed-rules train_bert_classifier

## Run evaluation. Usage: just test [config] [profile]
test c=config p=local:
    uv run snakemake test_all --workflow-profile {{p}} --configfile {{c}}

## Sync training results from VACC cluster
sync:
    ./tools/sync_vacc.sh

## Unlock snakemake directory if a process crashed
unlock:
    uv run snakemake --unlock

## Run local Quarto preview
preview:
    cd notebooks/quarto && quarto preview . --no-browser
