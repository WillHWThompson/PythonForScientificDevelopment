import pandas as pd
import yaml
from pathlib import Path
from src.core.schema import NLPExperimentConfig

def repair():
    results_dir = Path('results')
    config_path = Path('configs/nlp_baseline.yaml')
    
    with open(config_path) as f:
        base_cfg_data = yaml.safe_load(f)

    print("--- Starting Data Repair ---")
    for p in results_dir.glob('**/test_metrics.parquet'):
        folder = p.parent
        # Parse overrides from folder path (e.g., adapter_dim~256/learning_rate~0.0001)
        overrides = {}
        for part in folder.parts:
            if '~' in part:
                k, v = part.split('~')
                overrides[k] = v
        
        # Build fresh config with overrides
        cfg_data = base_cfg_data.copy()
        if 'model' not in cfg_data:
            cfg_data['model'] = {}
        if 'adapter_dim' in overrides:
            cfg_data['model']['adapter_dim'] = int(overrides['adapter_dim'])
            
        if 'training' not in cfg_data:
            cfg_data['training'] = {}
        if 'learning_rate' in overrides:
            cfg_data['training']['learning_rate'] = float(overrides['learning_rate'])
            
        config = NLPExperimentConfig(**cfg_data)
        correct_name = config.full_run_name
        
        print(f"Repairing: {folder} -> {correct_name}")
        
        # Fix Test Metrics
        df = pd.read_parquet(p)
        df['full_run_name'] = correct_name
        df.to_parquet(p)
        
        # Fix ROC Curves
        roc_p = folder / 'test_roc_curves.parquet'
        if roc_p.exists():
            df_roc = pd.read_parquet(roc_p)
            df_roc['full_run_name'] = correct_name
            df_roc.to_parquet(roc_p)
            
    print("--- Repair Complete ---")

if __name__ == "__main__":
    repair()
