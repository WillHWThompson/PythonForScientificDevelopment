# workflow/rules/data.smk

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
