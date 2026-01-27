import duckdb
from pydantic import BaseModel, ConfigDict
from typing import List

# This shows the "Vibecoding" power: 
# Querying a folder of flat files as if they were a SQL table,
# then validating the results with Pydantic.

class TrainingStat(BaseModel):
    """
    Pydantic model that matches our training output schema.
    Used to validate the data coming out of the Parquet files.
    """
    model_config = ConfigDict(from_attributes=True)
    
    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    f1_weighted: float
    full_run_name: str
    learning_rate: float

def query_results_directory(results_dir: str = "results/**/training_stats.parquet") -> List[TrainingStat]:
    """
    Queries all Parquet files in the results directory using DuckDB.
    """
    print(f"🔍 Searching for results in: {results_dir}")
    
    # DuckDB can treat a glob pattern as a table
    query = f"""
    SELECT 
        epoch, 
        train_loss, 
        val_loss, 
        accuracy, 
        f1_weighted, 
        full_run_name,
        learning_rate
    FROM read_parquet('{results_dir}')
    WHERE accuracy > 0.6  -- We can filter directly in DuckDB
    ORDER BY accuracy DESC
    """
    
    # Run the query and get a DataFrame (or list of dicts)
    df = duckdb.query(query).to_df()
    
    # Convert rows to Pydantic objects for type-safety/validation
    stats = [TrainingStat.model_validate(row) for _, row in df.iterrows()]
    return stats

if __name__ == "__main__":
    try:
        results = query_results_directory()
        
        print(f"\n✅ Successfully retrieved {len(results)} valid result records.\n")
        
        # Now we have full IDE support and validation for our data
        for i, res in enumerate(results[:5]):
            print(f"Result {i+1}:")
            print(f"  🏆 Run: {res.full_run_name}")
            print(f"  📈 Acc: {res.accuracy:.4f} | Loss: {res.train_loss:.4f}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error querying results: {e}")
        print("\nTip: Make sure you have run at least one experiment to generate Parquet files.")
