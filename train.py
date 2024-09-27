import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from Model.bert_model import BertDetector
from dataset import CustomDataset
from tqdm import tqdm
from sklearn.metrics import classification_report


def main():
    model_name = "bert-base-uncased"
    data_path = "Data"
    dataset_name = "MPQA"
    batch_size = 8
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5
    labels = {"text": "sentence", "label":"answer"}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertDetector(model_name)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    dataset_path = os.path.join(data_path, dataset_name)

    train_dataset = CustomDataset(dataset_path, "train", labels, tokenizer)
    val_dataset = CustomDataset(dataset_path, "val", labels, tokenizer)
    test_dataset = CustomDataset(dataset_path, "test", labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_datalodaer = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            tokenized, _, labels = batch
            labels = torch.tensor(labels).to(device)
            input_ids = tokenized["input_ids"].squeeze(1).to(device)       # [batch_size, 1, 512] -->  [batch_size, 512]
            attention_mask = tokenized["attention_mask"].squeeze(1).to(device) # [batch_size, 1, 512] -->  [batch_size, 512]
      
            optimizer.zero_grad()
            outputs = model((input_ids, attention_mask))  
            loss = criterion(outputs.float(), labels.unsqueeze(1).float())
            loss.backward()

            optimizer.step()

        if epoch % 2 == 0:
            model.eval()
            y_pred = []
            y_true = []
            for batch in tqdm(val_datalodaer):
                with torch.no_grad():
                    tokenized, _, labels = batch
                    labels = torch.tensor(labels).to(device)
                    input_ids = tokenized["input_ids"].squeeze(1).to(device)
                    attention_mask = tokenized["attention_mask"].squeeze(1).to(device)
                    
                    outputs = model((input_ids, attention_mask))
                    probabilities = torch.sigmoid(outputs)
                    preds = torch.where(probabilities > 0.5, torch.tensor(1.0), torch.tensor(0.0))
                    y_pred.extend(preds.cpu())
                    y_true.extend(labels.squeeze(-1).detach().cpu())

            print("VALIDATION PERFORMANCE: \n")
            print(classification_report(y_true, y_pred, target_names=["obj", "subj"]))

    
if __name__ == "__main__":

    main()
            

