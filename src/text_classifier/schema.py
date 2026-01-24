from pydantic import BaseModel, Field
from typing import List, Optional

class NLPModelConfig(BaseModel):
    """Configuration for the BERT + Adapter model."""
    model_name: str = Field(default="distilbert-base-uncased", description="Hugging Face model checkpoint.")
    adapter_dim: int = Field(default=256, description="Dimension of the bottleneck adapter layer.")
    num_classes: int = Field(default=4, description="Number of target classes (AG News has 4).")
    dropout: float = Field(default=0.1, description="Dropout rate for the classifier head.")

class NLPTrainingConfig(BaseModel):
    """Configuration for the PyTorch training process."""
    learning_rate: float = Field(default=5e-5, description="Initial learning rate for the adapter.")
    batch_size: int = Field(default=32, description="Batch size for training.")
    epochs: int = Field(default=3, description="Number of training epochs.")
    max_length: int = Field(default=128, description="Maximum sequence length for tokenization.")
    seed: int = Field(default=42, description="Random seed.")

class NLPDataConfig(BaseModel):
    """Configuration for data handling using datasets."""
    dataset_name: str = Field(default="ag_news", description="Hugging Face dataset identifier.")
    train_split: str = Field(default="train", description="Name of the training split.")
    val_split: str = Field(default="test", description="Name of the validation split.")

class NLPExperimentConfig(BaseModel):
    """Top-level configuration for a BERT classification experiment."""
    name: str = Field(..., description="Unique name for the experiment.")
    model: NLPModelConfig = Field(default_factory=NLPModelConfig)
    training: NLPTrainingConfig = Field(default_factory=NLPTrainingConfig)
    data: NLPDataConfig = Field(default_factory=NLPDataConfig)
    wandb_enabled: bool = Field(default=True, description="Enable WandB tracking.")
