# workflow/rules/train.smk

rule train_higgs_model:
    input:
        "data/higgs.parquet"
    output:
        params = "results/{cfg_name}/{exp_id}/trained_model.params",
        stats = "results/{cfg_name}/{exp_id}/training_stats.json"
    wildcard_constraints:
        cfg_name="[^/]+",
        exp_id="[^/]+"
    resources:
        mem_mb=16000,
        gpu=1,
        runtime="2h"
    run:
        # Use the config file provided on the CLI, or fallback to the one in configs/
        cfg_path = workflow.configfiles[0] if workflow.configfiles else f"configs/{wildcards.cfg_name}.yaml"
        
        shell(
            "python scripts/run_training.py "
            "--exp_name {wildcards.exp_id} "
            "--config {cfg_path} "
            "--output {output.params} "
            "--stats {output.stats}"
        )
