import argparse
import pandas as pd
import yaml
import json
import torch
from pathlib import Path
from src.core.schema import NLPExperimentConfig
from src.data.loader import TextClassifierDataLoader
from src.models.bert_adapter import BertAdapterClassifier
from src.core.trainer import TextClassifierTrainer

def main():
    parser = argparse.ArgumentParser(description="Train BERT Adapter Classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Path to save model weights")
    parser.add_argument("--stats", type=str, required=True, help="Path to save training stats (parquet)")
    parser.add_argument("--roc", type=str, required=True, help="Path to save ROC curve data (parquet)")
    
    # ... (parser arguments same as before)
    parser.add_argument("--adapter_dim", type=int, help="Override adapter dimension")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--group", type=str, help="WandB group identifier for clustering runs")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 2. Apply Overrides
    if args.adapter_dim:
        if 'model' not in config_data: config_data['model'] = {}
        config_data['model']['adapter_dim'] = args.adapter_dim
    if args.learning_rate:
        if 'training' not in config_data: config_data['training'] = {}
        config_data['training']['learning_rate'] = args.learning_rate
        
    config = NLPExperimentConfig(**config_data)

    print(f"Starting NLP Experiment: {config.full_run_name}")

    # 3. Setup WandB
    if config.wandb_enabled:
        import wandb
        wandb.init(
            project="bert-news-classification",
            name=config.full_run_name,
            group=args.group if args.group else config.name,
            config=config.model_dump()
        )

    # 4. Prepare Data
    data_manager = TextClassifierDataLoader(
        data_config=config.data, 
        train_config=config.training, 
        model_name=config.model.model_name
    )
    train_loader, val_loader = data_manager.get_dataloaders()

    # 5. Initialize Model
    model = BertAdapterClassifier(config.model)

    # 6. Train
    trainer = TextClassifierTrainer(config, model, train_loader, val_loader)
    history, roc_data = trainer.train()

    # 7. Save Results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    
    # Save training history
    df_history = pd.DataFrame(history)
    for k, v in {"exp_name": config.name, "full_run_name": config.full_run_name}.items():
        df_history[k] = v
    df_history.to_parquet(args.stats)

    # Save ROC curve data
    df_roc = pd.DataFrame(roc_data)
    df_roc["full_run_name"] = config.full_run_name
    df_roc.to_parquet(args.roc)

    if config.wandb_enabled:
        wandb.finish()

    print(f"Workflow Complete. Results saved to {args.stats} and {args.roc}")

if __name__ == "__main__":
    main()
