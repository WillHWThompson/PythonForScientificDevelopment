import argparse
import json
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import yaml
from src.scientific_dev.schema import ExperimentConfig, DataConfig
from src.scientific_dev.jax_model import (
    init_mlp_params, train_step, evaluate_model, mlp_forward
)
from src.scientific_dev.data_manager import HiggsDataManager

def main():
    parser = argparse.ArgumentParser(description="Train Higgs Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Path to save params")
    parser.add_argument("--stats", type=str, required=True, help="Path to save stats")
    
    # Overrides for Hyperparameter Sweeps
    parser.add_argument("--hidden_dims", type=str, help="Override hidden layers (e.g., '256-128')")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    config = ExperimentConfig(**config_data)

    # Apply Overrides from CLI (if present)
    if args.hidden_dims:
        config.model.hidden_dims = [int(x) for x in args.hidden_dims.split("-")]
        config.name += f"_hd_{args.hidden_dims}"
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        config.name += f"_lr_{args.learning_rate}"

    # Initialize Environment
    key = jax.random.PRNGKey(config.training.seed)
    data_manager = HiggsDataManager(config.data)
    
    # Init Parameters
    input_dim = 28 # Fixed for Higgs
    params = init_mlp_params(config.model.hidden_dims, input_dim, key)
    
    optimizer = optax.adam(config.training.learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize WandB
    if config.wandb.enabled:
        import wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.name,
            config=config.model_dump()
        )

    # Training Loop
    print(f"Starting Experiment: {config.name}")
    history = []
    
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Stream from DuckDB
        for X_batch, y_batch in data_manager.stream_batches(config.training.batch_size):
            params, opt_state, loss = train_step(params, opt_state, X_batch, y_batch, optimizer)
            epoch_loss += float(loss)
            batch_count += 1
            
            if batch_count % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"--- Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f} ---")
        
        metrics = {"epoch": epoch, "loss": avg_loss}
        history.append(metrics)
        
        if config.wandb.enabled:
            wandb.log(metrics)

    # Save Results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, 'w') as f:
        json.dump({
            "config": config.model_dump(),
            "history": history
        }, f, indent=2)
    
    # Mock save for params
    with open(args.output, 'w') as f:
        f.write("OPTIMIZED_PARAMS_PLACEHOLDER")

    if config.wandb.enabled:
        wandb.finish()

    print(f"Training finished. Results saved to {args.stats}")

if __name__ == "__main__":
    main()
