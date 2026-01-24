# workflow/rules/train.smk

rule train_bert_classifier:
    output:
        weights = "results/{cfg_name}/{params_path}/model.pt",
        stats = "results/{cfg_name}/{params_path}/training_stats.parquet"
    wildcard_constraints:
        cfg_name="[^/]+",
        params_path=".+"
    resources:
        mem_mb=16000,
        gpu=1,
        runtime="4h"
    params:
        # Resolve config logic in params to keep shell command clean
        cfg = lambda w: workflow.configfiles[0] if workflow.configfiles else f"configs/{w.cfg_name}.yaml",
        # Parse overrides from the params_path wildcard
        overrides = lambda w: " ".join([f"--{p.split('~')[0]} {p.split('~')[1]}" for p in w.params_path.split("/") if "~" in p]) if w.params_path != "base" else ""
    shell:
        "PYTHONPATH=$(pwd):$(pwd)/src ./.venv/bin/python scripts/run_training.py "
        "--config {params.cfg} "
        "--output {output.weights} "
        "--stats {output.stats} "
        "{params.overrides}"
