import argparse
import json
import torch
import yaml
from pathlib import Path
from src.text_classifier.schema import NLPExperimentConfig
from src.text_classifier.data_loader import TextClassifierDataLoader
from src.text_classifier.model import BertAdapterClassifier
from src.text_classifier.trainer import TextClassifierTrainer

def main():
    parser = argparse.ArgumentParser(description="Train BERT Text Classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Path to save model weights")
    parser.add_argument("--stats", type=str, required=True, help="Path to save training stats")
    
    # Sweep Overrides
    parser.add_argument("--adapter_dim", type=int, help="Override adapter dimension")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    # 2. Integrate Overrides directly into the ingestion
    if args.adapter_dim:
        config_data['model']['adapter_dim'] = args.adapter_dim
    if args.learning_rate:
        config_data['training']['learning_rate'] = args.learning_rate
        
    config = NLPExperimentConfig(**config_data)
    
    print(f"Starting NLP Experiment: {config.full_run_name}")

    # 3. Setup WandB
    if config.wandb_enabled:
        import wandb
        wandb.init(
            project="bert-news-classification",
            name=config.full_run_name,
            config=config.model_dump()
        )

    # 4. Prepare Data
    data_manager = TextClassifierDataLoader(config.data, config.training, config.model.model_name)
    train_loader, val_loader = data_manager.get_dataloaders()

    # 5. Initialize Model
    model = BertAdapterClassifier(config.model)

    # 6. Train
    trainer = TextClassifierTrainer(config, model, train_loader, val_loader)
    history = trainer.train()

    # 7. Save Results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # We'll just save the adapter weights to be "Parameter Efficient"
    # Actually for simplicity in this demo, let's just save the state dict
    torch.save(model.state_dict(), args.output)
    
    with open(args.stats, 'w') as f:
        json.dump({
            "config": config.model_dump(),
            "history": history
        }, f, indent=2)

    if config.wandb_enabled:
        wandb.finish()

    print(f"Workflow Complete. Model saved to {args.output}")

if __name__ == "__main__":
    main()
