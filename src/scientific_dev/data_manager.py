import duckdb
import pandas as pd
import jax.numpy as jnp
import json
from pathlib import Path
from .schema import DataConfig

class HiggsDataManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.con = duckdb.connect(database=':memory:')
        
    def prepare_data(self):
        """Discovers or prepares the Parquet data."""
        if not Path(self.config.parquet_path).exists():
            print(f"Parquet file not found at {self.config.parquet_path}.")
            print("In a real workflow, Snakemake would have generated this.")
            return False
        return True

    def get_summary_stats(self):
        """Uses DuckDB to calculate summary statistics across the 11M rows."""
        query = f"""
            SELECT 
                column0 as label,
                avg(column1) as avg_lepton_pt,
                stddev(column1) as std_lepton_pt,
                count(*) as count
            FROM read_parquet('{self.config.parquet_path}')
            GROUP BY label
        """
        return self.con.execute(query).df()

    def stream_batches(self, batch_size: int, split: str = "train"):
        """
        An efficient generator that streams batches from Parquet via DuckDB.
        Avoids loading the entire 11M rows into memory.
        """
        # Note: HIGGS.csv has no header in the original dataset. 
        # column0 is the label.
        
        # In a real setup, we'd have a column for 'split'
        query = f"SELECT * FROM read_parquet('{self.config.parquet_path}')"
        
        # We can use duckdb's fetch_df_chunk() for streaming
        res = self.con.execute(query)
        while True:
            chunk = res.fetch_df_chunk(batch_size)
            if chunk.empty:
                break
            
            X = chunk.iloc[:, 1:].values.astype(jnp.float32)
            y = chunk.iloc[:, 0].values.astype(jnp.float32)
            yield jnp.array(X), jnp.array(y)

    def aggregate_sweep_results(self, results_dir: str):
        """
        Uses DuckDB to aggregate all training_stats.json files from a sweep
        into a single flattened DataFrame for analysis.
        """
        results_path = Path(results_dir)
        json_pattern = str(results_path / "**" / "training_stats.json")
        
        # DuckDB can read a list of JSON files directly and flatten them
        query = f"""
            SELECT 
                config.name as exp_name,
                config.model.hidden_dims as architecture,
                config.training.learning_rate as lr,
                config.training.epochs as epochs,
                history as training_history
            FROM read_json_auto('{json_pattern}')
        """
        return self.con.execute(query).df()

def convert_csv_to_parquet(csv_path: str, parquet_path: str):
    """Utility to convert the massive CSV to Parquet using DuckDB."""
    con = duckdb.connect()
    # In HIGGS dataset, first col is label, next 28 are features
    print(f"Converting {csv_path} to {parquet_path}...")
    con.execute(f"COPY (SELECT * FROM read_csv_auto('{csv_path}')) TO '{parquet_path}' (FORMAT PARQUET)")
    print("Conversion complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Higgs Data Manager CLI")
    parser.add_argument("--csv", type=str, help="Path to raw CSV")
    parser.add_argument("--parquet", type=str, help="Path to output Parquet")
    args = parser.parse_args()

    if args.csv and args.parquet:
        convert_csv_to_parquet(args.csv, args.parquet)
    else:
        # Default demo behavior
        config = DataConfig(parquet_path="data/higgs.parquet")
        manager = HiggsDataManager(config)
        if manager.prepare_data():
            stats = manager.get_summary_stats()
            print("Summary Statistics from DuckDB:")
            print(stats)
