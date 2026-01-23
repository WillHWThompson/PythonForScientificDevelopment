from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    """Configuration for the MLP architecture."""
    hidden_dims: List[int] = Field(default=[256, 128, 64], description="List of hidden layer dimensions.")
    activation: str = Field(default="relu", description="Activation function to use.")
    dropout_rate: float = Field(default=0.0, description="Dropout rate for regularization.")

class TrainingConfig(BaseModel):
    """Configuration for the JAX training process."""
    learning_rate: float = Field(default=1e-3, description="Initial learning rate.")
    batch_size: int = Field(default=1024, description="Training batch size.")
    epochs: int = Field(default=10, description="Number of training epochs.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")

class DataConfig(BaseModel):
    """Configuration for data handling."""
    raw_csv_url: str = Field(
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        description="URL to the raw Higgs dataset."
    )
    parquet_path: str = Field(default="data/higgs.parquet", description="Path to store the processed Parquet file.")
    use_high_level_features: bool = Field(default=True, description="Whether to include the 7 high-level features.")

class WandBConfig(BaseModel):
    """Configuration for Weights & Biases tracking."""
    enabled: bool = Field(default=False, description="Whether to enable WandB tracking.")
    project: str = Field(default="higgs-classification", description="WandB project name.")
    entity: Optional[str] = Field(default=None, description="WandB entity/username.")
    group: Optional[str] = Field(default=None, description="WandB run group.")

class ExperimentConfig(BaseModel):
    """Top-level configuration for a Higgs classification experiment."""
    name: str = Field(..., description="Unique name for the experiment.")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)

if __name__ == "__main__":
    # Quick demo of validation
    config = ExperimentConfig(name="baseline_mlp")
    print(config.model_dump_json(indent=2))
