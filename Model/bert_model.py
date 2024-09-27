from transformers import BertModel
import torch.nn as nn


class BertDetector(nn.Module):
    def __init__(self, 
                 model_name="bert-base-uncased", 
                 dropout=0.4,
                 output_dim=1):
        super(BertDetector, self).__init__()
        self.bert_layer = BertModel.from_pretrained(model_name)
        self.dropout_layer = nn.Dropout(dropout)
        self.ff = nn.Linear(self.bert_layer.config.hidden_size, output_dim)

    def forward(self, x):
        input_ids = x[0]
        attention_mask = x[1]
        x = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = x.pooler_output
        x = self.dropout_layer(pooled_output)
        output = self.ff(x)

        return output