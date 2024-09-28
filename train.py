import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from Model.bert_model import BertDetector
from dataset import CustomDataset
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, f1_score


def main(args):
    model_name = "bert-base-uncased"
    data_path = "Data"
    model_save_path = "Model"
    dataset_name = args.dataset_name
    batch_size = 8
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    mpqa_labels = {"text": "sentence", "label":"answer"}
    news_1_labels = {"text": "Sentence", "label": "Label"}
    news_2_labels = {"text": "text", "label": "labels"}

    # map labels to dataset columns
    if dataset_name == "MPQA":
        labels = mpqa_labels
    elif dataset_name == "News-1":
        labels = news_1_labels
    elif dataset_name == "News-2":
        labels = news_2_labels

    if dataset_name != "all":
        dataset_path = os.path.join(data_path, dataset_name)
        train_dataset = CustomDataset(dataset_path, "train", labels, tokenizer)
        val_dataset = CustomDataset(dataset_path, "val", labels, tokenizer)

    elif dataset_name == "all":
        mpqa_dataset_path = os.path.join(data_path, "MPQA")
        news_1_dataset_path = os.path.join(data_path, "News-1")
        news_2_dataset_path = os.path.join(data_path, "News-2")

        mpqa_train_dataset = CustomDataset(mpqa_dataset_path, "train", mpqa_labels, tokenizer)
        news_1_train_dataset = CustomDataset(news_1_dataset_path, "train", news_1_labels, tokenizer)
        news_2_train_dataset = CustomDataset(news_2_dataset_path, "train", news_2_labels, tokenizer)

        mpqa_val_dataset = CustomDataset(mpqa_dataset_path, "val", mpqa_labels, tokenizer)
        news_1_val_dataset = CustomDataset(news_1_dataset_path, "val", news_1_labels, tokenizer)
        news_2_val_dataset = CustomDataset(news_2_dataset_path, "val", news_2_labels, tokenizer)
        # combining all the datasets
        train_dataset = ConcatDataset([mpqa_train_dataset, news_1_train_dataset, news_2_train_dataset])
        val_dataset = ConcatDataset([mpqa_val_dataset, news_1_val_dataset, news_2_val_dataset])

    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_datalodaer = DataLoader(val_dataset, batch_size, shuffle=True)

    model = BertDetector(model_name)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # track best model on validation
    best_val_f1_score = 0
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
                    y_pred.extend(preds.cpu().numpy())
                        
                    y_true.extend(labels.cpu().numpy().tolist())

            print("VALIDATION PERFORMANCE: \n")
            print(classification_report(y_true, y_pred, target_names=["obj", "subj"]))

            val_f1_score = f1_score(y_true, y_pred, average='macro')
            if val_f1_score > best_val_f1_score:
                best_val_f1_score = val_f1_score
                best_model_weights = model.state_dict()

    
    if best_model_weights is not None:
        torch.save(best_model_weights, os.path.join(model_save_path, dataset_name)+"best_model.pth")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset to train and test on. To test on all datasets type 'all'")

    args = parser.parse_args()
    main(args)
            

