# Default Paths
config := "configs/nlp_baseline.yaml"

## Run full pipeline. Usage: just all [profile] [config]
all p="local" c=config:
    uv run snakemake --workflow-profile ./workflow/profiles/{{p}} --configfile {{c}}

## Run training. Usage: just train [profile] [config]
train p="local" c=config:
    uv run snakemake train_all --workflow-profile ./workflow/profiles/{{p}} --configfile {{c}}

## Run evaluation. Usage: just test [profile] [config]
test p="local" c=config:
    uv run snakemake test_all --workflow-profile ./workflow/profiles/{{p}} --configfile {{c}}

## Sync training results from VACC cluster
sync:
    ./tools/sync_vacc.sh

## Regenerate figures locally using synced data
plot:
    uv run snakemake plots/fig1_roc_sweep.png -j1

## Unlock snakemake directory if a process crashed
unlock:
    uv run snakemake --unlock

## Print the command to activate the environment
activate:
    @echo "Run this command to activate your environment:"
    @echo "source .venv/bin/activate"

## Run local Quarto preview
preview:
    cd notebooks/quarto && quarto preview . --no-browser
