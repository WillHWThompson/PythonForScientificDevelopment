# Snakemake Pipeline for Higgs Boson Classification

configfile: "config.yaml"

rule all:
    input:
        "results/trained_model.params",
        "results/training_stats.json"

rule download_data:
    output:
        "data/HIGGS.csv.gz"
    shell:
        "wget -O {output} https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

rule decompress_data:
    input:
        "data/HIGGS.csv.gz"
    output:
        temp("data/HIGGS.csv")
    shell:
        "gunzip -c {input} > {output}"

rule ingest_to_parquet:
    input:
        "data/HIGGS.csv"
    output:
        "data/higgs.parquet"
    resources:
        mem_mb=8000
    shell:
        "python -m src.scientific_dev.data_manager --csv {input} --parquet {output}"

rule train_higgs_model:
    input:
        "data/higgs.parquet"
    output:
        params = "results/trained_model.params",
        stats = "results/training_stats.json"
    resources:
        mem_mb=16000,
        gpu=1,
        runtime="2h"
    shell:
        "python scripts/run_training.py --config config.yaml --output {output.params} --stats {output.stats}"
