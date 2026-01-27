from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import Any

from ..core.schema import NLPDataConfig, NLPTrainingConfig

class TextClassifierDataLoader(BaseModel):
    """
    Pydantic-validated data loader for text classification.
    Manages dataset loading, tokenization, and PyTorch DataLoader creation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_config: NLPDataConfig = Field(..., description="Configuration for dataset splits and limits.")
    train_config: NLPTrainingConfig = Field(..., description="Configuration for batching and sequence length.")
    model_name: str = Field(..., description="Hugging Face model checkpoint for the tokenizer.")
    
    # Use PrivateAttr for external objects that shouldn't be part of the schema
    _tokenizer: Any = PrivateAttr()

    @property
    def tokenizer(self) -> Any:
        """Expose the internal tokenizer."""
        return self._tokenizer

    def model_post_init(self, __context: Any) -> None:
        """Initialize the tokenizer after the model is validated."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def prepare_datasets(self):
        """Loads and tokenizes the dataset."""
        dataset = load_dataset(self.data_config.dataset_name)
        
        # Subsetting for fast local test
        if self.data_config.max_samples:
            limit = self.data_config.max_samples
            for split in dataset.keys():
                actual_limit = min(len(dataset[split]), limit)
                dataset[split] = dataset[split].select(range(actual_limit))

        def tokenize_function(examples):
            return self._tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=self.train_config.max_length
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # Format for PyTorch
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets

    def get_dataloaders(self):
        """Returns training and evaluation (validation) DataLoaders."""
        datasets = self.prepare_datasets()
        
        train_loader = DataLoader(
            datasets[self.data_config.train_split], 
            batch_size=self.train_config.batch_size, 
            shuffle=True
        )
        eval_loader = DataLoader(
            datasets[self.data_config.eval_split], 
            batch_size=self.train_config.batch_size
        )
        
        return train_loader, eval_loader

    def get_test_loader(self):
        """Returns the final test DataLoader."""
        datasets = self.prepare_datasets()
        test_loader = DataLoader(
            datasets[self.data_config.test_split],
            batch_size=self.train_config.batch_size
        )
        return test_loader
