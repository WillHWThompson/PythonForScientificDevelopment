import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import evaluate
from ..core.schema import NLPExperimentConfig

class TextClassifierTrainer:
    def __init__(self, config: NLPExperimentConfig, model: nn.Module, train_loader=None, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training setup only if train_loader is provided
        if train_loader is not None:
            self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate)
            
            # Scheduler
            num_training_steps = len(train_loader) * config.training.epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(0.1 * num_training_steps), 
                num_training_steps=num_training_steps
            )
        else:
            self.optimizer = None
            self.scheduler = None
        
        # Metrics
        self.acc_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for step, batch in enumerate(progress_bar):
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
            # Real-time step logging
            if self.config.wandb_enabled and step % self.config.training.log_interval == 0:
                import wandb
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/global_step": epoch * len(self.train_loader) + step
                })
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader=None, name="Eval", epoch=None):
        self.model.eval()
        total_loss = 0
        eval_loader = loader if loader is not None else self.val_loader
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"{'Epoch ' + str(epoch) if epoch is not None else ''} [{name}]"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                self.acc_metric.add_batch(predictions=preds, references=labels)
                self.f1_metric.add_batch(predictions=preds, references=labels)
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
        metrics = self.acc_metric.compute()
        f1_metrics = self.f1_metric.compute(average="weighted")
        metrics.update(f1_metrics)
        metrics["avg_loss"] = total_loss / len(eval_loader)

        # Advanced Metrics: AUROC & ROC Curves
        import numpy as np
        from sklearn.metrics import roc_auc_score, roc_curve
        from torch.nn.functional import softmax
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).numpy()
        probs = softmax(all_logits, dim=1).numpy()
        
        roc_data = [] # For plotting curves
        try:
            metrics["auroc"] = roc_auc_score(all_labels, probs, multi_class="ovr", average="weighted")
            
            # Generate ROC curves for each class (OVR)
            label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            for i in range(4):
                fpr, tpr, _ = roc_curve(all_labels == i, probs[:, i])
                # Downsample curve for efficient storage and plotting
                indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
                for idx in indices:
                    roc_data.append({
                        "class": label_names[i],
                        "fpr": float(fpr[idx]),
                        "tpr": float(tpr[idx])
                    })
        except Exception as e:
            print(f"Warning: ROC computation failed: {e}")
            metrics["auroc"] = 0.0
        
        return metrics, roc_data

    def test(self, test_loader):
        """Final independent model testing with full diagnostics."""
        print("--- Running Final Test Evaluation ---")
        metrics, roc_data = self.evaluate(loader=test_loader, name="Test")
        
        test_data = {
            "test/loss": metrics["avg_loss"],
            "test/accuracy": metrics["accuracy"],
            "test/f1_weighted": metrics["f1"],
            "test/auroc": metrics["auroc"]
        }
        
        print(f"Test Loss: {metrics['avg_loss']:.4f} | Test Acc: {metrics['accuracy']:.4f} | AUROC: {metrics['auroc']:.4f}")
        
        if self.config.wandb_enabled:
            import wandb
            wandb.log(test_data)
            
        return test_data, roc_data

    def train(self):
        history = []
        
        last_roc_data = []
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch(epoch)
            metrics, last_roc_data = self.evaluate(epoch=epoch)
            
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
                
        return history, last_roc_data
