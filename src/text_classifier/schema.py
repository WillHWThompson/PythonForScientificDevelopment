from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

class NLPModelConfig(BaseModel):
    """Configuration for the BERT + Adapter model."""
    model_name: str = Field(default="distilbert-base-uncased", description="Hugging Face model checkpoint.")
    adapter_dim: int = Field(default=256, description="Dimension of the bottleneck adapter layer.")
    num_classes: int = Field(default=4, description="Number of target classes (AG News has 4).")
    dropout: float = Field(default=0.1, description="Dropout rate for the classifier head.")

    @field_validator('adapter_dim')
    @classmethod
    def validate_adapter_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("adapter_dim must be a positive integer")
        return v

class NLPTrainingConfig(BaseModel):
    """Configuration for the PyTorch training process."""
    learning_rate: float = Field(default=5e-5, description="Initial learning rate for the adapter.")
    batch_size: int = Field(default=32, description="Batch size for training.")
    epochs: int = Field(default=3, description="Number of training epochs.")
    max_length: int = Field(default=128, description="Maximum sequence length for tokenization.")
    seed: int = Field(default=42, description="Random seed.")

    @field_validator('learning_rate')
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

class NLPDataConfig(BaseModel):
    """Configuration for data handling using datasets."""
    dataset_name: str = Field(default="ag_news", description="Hugging Face dataset identifier.")
    train_split: str = Field(default="train", description="Name of the training split.")
    val_split: str = Field(default="test", description="Name of the validation split.")

class NLPExperimentConfig(BaseModel):
    """Top-level configuration for a BERT classification experiment."""
    name: str = Field(default="bert_news_experiment", description="Base name for the experiment.")
    model: NLPModelConfig = Field(default_factory=NLPModelConfig)
    training: NLPTrainingConfig = Field(default_factory=NLPTrainingConfig)
    data: NLPDataConfig = Field(default_factory=NLPDataConfig)
    wandb_enabled: bool = Field(default=True, description="Enable WandB tracking.")
    
    # This field will be automatically updated by the validator
    full_run_name: str = Field(default="", description="Automatically generated name including hyperparameters.")

    @model_validator(mode='after')
    def generate_full_name(self) -> 'NLPExperimentConfig':
        """Automatically generates a descriptive experiment name from parameters."""
        suffix = f"adim_{self.model.adapter_dim}_lr_{self.training.learning_rate}"
        self.full_run_name = f"{self.name}_{suffix}"
        return self
