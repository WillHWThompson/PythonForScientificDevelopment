import yaml
import pandas as pd
from itertools import product
from pathlib import Path
from datetime import datetime

# 1. Determine Workflow Mode and Build Experiment Grid
config_file_path = workflow.configfiles[0] if workflow.configfiles else "configs/nlp_baseline.yaml"

# Load config to get the name and sweep info
with open(config_file_path, 'r') as f:
    config_data = yaml.safe_load(f)
cfg_name = Path(config_file_path).stem

# Create a unique group ID for WandB grouping
RUN_GROUP_ID = f"{cfg_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"

TRAIN_OUTPUTS = []
TEST_OUTPUTS = []

def get_instance_dir(row_dict):
    """Helper to convert a parameter dict into a directory path: param1~val1/param2~val2"""
    parts = []
    for k, v in row_dict.items():
        parts.append(f"{k}~{v}")
    return "/".join(parts)

if "sweep" in config_data:
    keys = config_data["sweep"].keys()
    values = config_data["sweep"].values()
    
    # Generate all combinations
    grid = [dict(zip(keys, v)) for v in product(*values)]
    
    # Build output paths
    for row in grid:
        instance_path = get_instance_dir(row)
        TRAIN_OUTPUTS.append(f"results/{cfg_name}/{instance_path}/training_stats.parquet")
        TRAIN_OUTPUTS.append(f"results/{cfg_name}/{instance_path}/roc_curves.parquet")
        TRAIN_OUTPUTS.append(f"results/{cfg_name}/{instance_path}/model.pt")
        
        TEST_OUTPUTS.append(f"results/{cfg_name}/{instance_path}/test_metrics.parquet")
        TEST_OUTPUTS.append(f"results/{cfg_name}/{instance_path}/test_roc_curves.parquet")
else:
    TRAIN_OUTPUTS.append(f"results/{cfg_name}/base/training_stats.parquet")
    TRAIN_OUTPUTS.append(f"results/{cfg_name}/base/roc_curves.parquet")
    TRAIN_OUTPUTS.append(f"results/{cfg_name}/base/model.pt")
    
    TEST_OUTPUTS.append(f"results/{cfg_name}/base/test_metrics.parquet")
    TEST_OUTPUTS.append(f"results/{cfg_name}/base/test_roc_curves.parquet")

rule all:
    input:
        TRAIN_OUTPUTS + TEST_OUTPUTS

rule test_all:
    input:
        TEST_OUTPUTS

# 2. Include Modular Rules
include: "workflow/rules/train.smk"
include: "workflow/rules/test.smk"
