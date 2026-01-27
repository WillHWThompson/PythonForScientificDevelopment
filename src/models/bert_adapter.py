import torch.nn as nn
from transformers import AutoModel
from ..core.schema import NLPModelConfig

class BertAdapterClassifier(nn.Module):
    def __init__(self, config: NLPModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained BERT backbone
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Freeze BERT weights to demonstrate the "adapter" pattern
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Hidden dimension of DistilBERT is 768
        bert_hidden_dim = self.bert.config.hidden_size
        
        # Bottleneck Adapter structure (The "MLP" part)
        self.adapter = nn.Sequential(
            nn.Linear(bert_hidden_dim, config.adapter_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_dim, bert_hidden_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(bert_hidden_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        # 1. Get BERT representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use only the 0th token ([CLS]) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 2. Pass through bottleneck adapter (Our MLP)
        adapted_output = self.adapter(cls_output)
        
        # 3. Residual connection
        combined_output = cls_output + adapted_output
        
        # 4. Final classification
        # We return raw logits because nn.CrossEntropyLoss in PyTorch applies LogSoftmax internally,
        # which is more numerically stable than applying Softmax manually.
        logits = self.classifier(self.dropout(combined_output))
        
        return logits
