import argparse
import yaml
import pandas as pd
import torch
from pathlib import Path

from src.core.schema import NLPExperimentConfig
from src.models.bert_adapter import BERTWithAdapter
from src.data.loader import TextClassifierDataLoader
from src.core.trainer import TextClassifierTrainer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained BERT Adapter model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config YAML.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model.pt weights.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the test metrics (parquet).")
    parser.add_argument("--roc", type=str, required=True, help="Path to save the ROC curve data (parquet).")
    
    # Allow hyperparameter overrides similar to run_training.py
    parser.add_argument("--adapter_dim", type=int, help="Override adapter dimension.")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate.")
    
    args = parser.parse_args()

    # 1. Load and Override Config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = NLPExperimentConfig(**config_dict)
    if args.adapter_dim: config.model.adapter_dim = args.adapter_dim
    if args.learning_rate: config.training.learning_rate = args.learning_rate

    print(f"--- Evaluating Run: {config.full_run_name} ---")

    # 2. Initialize Model and Load Weights
    model = BertAdapterClassifier(config.model)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    
    # 3. Prepare Test Data
    data_loader = TextClassifierDataLoader(config.data, config.training, config.model.model_name)
    test_loader = data_loader.get_test_loader()
    
    # 4. Initialize Trainer (only for metrics logic)
    # We pass dummy loaders for train/eval since we only use the .test() method
    trainer = TextClassifierTrainer(config, model, train_loader=None, val_loader=None)
    
    # 5. Run Test
    test_metrics, roc_data = trainer.test(test_loader)
    
    # 6. Save Results
    # Flatten config and merge with metrics
    results = config.model.model_dump()
    results.update(config.training.model_dump())
    results.update(test_metrics)
    results["full_run_name"] = config.full_run_name
    
    df = pd.DataFrame([results])
    df.to_parquet(args.output)
    
    # Save ROC data
    df_roc = pd.DataFrame(roc_data)
    df_roc["full_run_name"] = config.full_run_name
    df_roc.to_parquet(args.roc)
    
    print(f"Test metrics saved to {args.output} and {args.roc}")

if __name__ == "__main__":
    main()
