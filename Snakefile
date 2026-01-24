# Snakefile
import yaml
from pathlib import Path

# 1. Include Modular Rules
include: "workflow/rules/data.smk"
include: "workflow/rules/train.smk"

# 2. Determine Workflow Mode
# We expect the user to run: snakemake --configfile configs/my_experiment.yaml
# The 'config' object is automatically populated by Snakemake from the YAML.

# Derive the config name from the filename provided to --configfile
# Default to "default" if no configfile is provided (though we'll error out if sweep info is missing)
config_file_path = workflow.configfiles[0] if workflow.configfiles else "configs/default.yaml"
cfg_name = Path(config_file_path).stem

ALL_OUTPUTS = []

if "sweep" in config:
    for h_dim in config["sweep"]["hidden_dims"]:
        for lr in config["sweep"]["learning_rate"]:
            dim_str = "_".join(map(str, h_dim))
            exp_id = f"mlp_{dim_str}_lr_{lr}"
            ALL_OUTPUTS.append(f"results/{cfg_name}/{exp_id}/training_stats.json")
            ALL_OUTPUTS.append(f"results/{cfg_name}/{exp_id}/trained_model.params")
else:
    # If no sweep is defined, just run the base model from the config
    ALL_OUTPUTS.append(f"results/{cfg_name}/base/training_stats.json")
    ALL_OUTPUTS.append(f"results/{cfg_name}/base/trained_model.params")

rule all:
    input:
        ALL_OUTPUTS
