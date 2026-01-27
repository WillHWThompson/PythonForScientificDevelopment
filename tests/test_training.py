from src.core.schema import NLPExperimentConfig
from src.data.loader import TextClassifierDataLoader
from src.models.bert_adapter import BertAdapterClassifier
from src.core.trainer import TextClassifierTrainer

def test_training_loss_decreases():
    """Integration Test: Verifies that training loss decreases over 2 epochs on a small subset."""
    config = NLPExperimentConfig()
    config.data.max_samples = 32
    config.training.epochs = 2
    config.training.batch_size = 8
    config.training.max_length = 32
    config.training.learning_rate = 1e-3  # High LR for quick movement
    config.wandb_enabled = False
    
    # Setup data
    loader_manager = TextClassifierDataLoader(
        data_config=config.data,
        train_config=config.training,
        model_name=config.model.model_name
    )
    train_loader, val_loader = loader_manager.get_dataloaders()
    
    # Setup model
    model = BertAdapterClassifier(config.model)
    
    # Setup trainer
    trainer = TextClassifierTrainer(config, model, train_loader, val_loader)
    
    # Train
    history, _ = trainer.train()
    
    assert len(history) == 2
    loss_0 = history[0]['train_loss']
    loss_1 = history[1]['train_loss']
    
    print(f"Epoch 0 Loss: {loss_0:.4f}, Epoch 1 Loss: {loss_1:.4f}")
    assert loss_1 < loss_0, f"Loss did not decrease: {loss_1} >= {loss_0}"
