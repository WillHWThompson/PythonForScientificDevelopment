import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import evaluate
import json
from pathlib import Path
from ..core.schema import NLPExperimentConfig

class TextClassifierTrainer:
    def __init__(self, config: NLPExperimentConfig, model: nn.Module, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        num_training_steps = len(train_loader) * config.training.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(0.1 * num_training_steps), 
            num_training_steps=num_training_steps
        )
        
        # Metrics
        self.acc_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)

    def evaluate(self, epoch=None):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch if epoch is not None else ''} [Eval]"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                self.acc_metric.add_batch(predictions=preds, references=labels)
                self.f1_metric.add_batch(predictions=preds, references=labels)
                
        metrics = self.acc_metric.compute()
        f1_metrics = self.f1_metric.compute(average="weighted")
        metrics.update(f1_metrics)
        metrics["avg_loss"] = total_loss / len(self.val_loader)
        
        return metrics

    def train(self):
        history = []
        
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate(epoch)
            
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["avg_loss"],
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1"]
            }
            history.append(epoch_data)
            
            print(f"--- Epoch {epoch} Summary ---")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['avg_loss']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f} | F1 (Weighted): {metrics['f1']:.4f}")
            
            if self.config.wandb_enabled:
                import wandb
                wandb.log(epoch_data)
                
        return history
