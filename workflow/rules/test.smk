# workflow/rules/test.smk

rule test_bert_classifier:
    input:
        weights = "results/{cfg_name}/{params_path}/model.pt"
    output:
        test_stats = "results/{cfg_name}/{params_path}/test_metrics.parquet",
        roc = "results/{cfg_name}/{params_path}/test_roc_curves.parquet"
    wildcard_constraints:
        cfg_name="[^/]+",
        params_path=".+"
    resources:
        mem_mb=8000,   # Testing requires less memory than training
        gpu=1,
        runtime="1h"
    params:
        cfg = lambda w: workflow.configfiles[0] if workflow.configfiles else f"configs/{w.cfg_name}.yaml",
        overrides = lambda w: " ".join([f"--{p.split('~')[0]} {p.split('~')[1]}" for p in w.params_path.split("/") if "~" in p]) if w.params_path != "base" else ""
    shell:
        "PYTHONPATH=$(pwd):$(pwd)/src ./.venv/bin/python scripts/run_testing.py "
        "--config {params.cfg} "
        "--weights {input.weights} "
        "--output {output.test_stats} "
        "--roc {output.roc} "
        "{params.overrides}"
