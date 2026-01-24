# workflow/rules/nlp_train.smk

rule train_bert_classifier:
    output:
        weights = "results/{cfg_name}/{params_path}/model.pt",
        stats = "results/{cfg_name}/{params_path}/training_stats.json"
    wildcard_constraints:
        cfg_name="[^/]+",
        params_path=".+"
    resources:
        mem_mb=16000,
        gpu=1,
        runtime="4h"
    run:
        # Use the config file provided on the CLI, or fallback to the one in configs/
        cfg_path = workflow.configfiles[0] if workflow.configfiles else f"configs/{wildcards.cfg_name}.yaml"
        
        # Build command args by parsing the params_path
        # format: param1~val1/param2~val2
        cmd_args = f"--config {cfg_path} --output {output.weights} --stats {output.stats} "
        
        # Extract individual parameters from the path
        if wildcards.params_path != "base":
            for part in wildcards.params_path.split("/"):
                if "~" in part:
                    k, v = part.split("~")
                    cmd_args += f"--{k} {v} "
        
        shell(f"python scripts/run_nlp_training.py {cmd_args}")
