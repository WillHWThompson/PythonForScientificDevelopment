import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from .schema import NLPDataConfig, NLPTrainingConfig

class TextClassifierDataLoader:
    def __init__(self, data_config: NLPDataConfig, train_config: NLPTrainingConfig, model_name: str):
        self.data_config = data_config
        self.train_config = train_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_datasets(self):
        """Loads and tokenizes the AG News dataset."""
        dataset = load_dataset(self.data_config.dataset_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
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
        """Returns training and validation DataLoaders."""
        datasets = self.prepare_datasets()
        
        train_loader = DataLoader(
            datasets[self.data_config.train_split], 
            batch_size=self.train_config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            datasets[self.data_config.val_split], 
            batch_size=self.train_config.batch_size
        )
        
        return train_loader, val_loader
