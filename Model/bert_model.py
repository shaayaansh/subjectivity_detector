from transformers import BertModel
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self, 
                 model_name="bert-base-uncased", 
                 dropout=0.4,
                 output_dim=2):
        super().__init__()
        self.bert_layer = BertModel.from_pretrained(model_name)
        self.dropout_layer = nn.Dropout(dropout)
        self.ff = nn.Linear(self.bert_layer.config.hidden_size, output_dim)

    def forward(self, x):
        x = self.bert_layer(x)
        x = self.dropout_layer(x)
        output = self.ff(x)

        return output