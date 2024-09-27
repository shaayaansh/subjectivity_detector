import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os


class CustomDataset(Dataset):
    def __init__(self, data_path,
                  batch_type, labels, tokenizer):
        super(CustomDataset, self).__init__()
        
        self.labels = labels
        self.tokenizer = tokenizer
        data_splits = [os.path.join(data_path, split)
                        for split in os.listdir(data_path)]
        

        if batch_type == "train":
            dataset_path = [dataset for dataset in data_splits if "train" in dataset][0]
            self.dataset = pd.read_csv(dataset_path)

        elif batch_type == "val":
            dataset_path = [dataset for dataset in data_splits if "val" in dataset][0]
            self.dataset = pd.read_csv(dataset_path)

        else:
            dataset_path = [dataset for dataset in data_splits if "test" in dataset][0]
            self.dataset = pd.read_csv(dataset_path)
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[self.labels["text"]].iloc[idx]
        label = self.dataset[self.labels["label"]].iloc[idx]
        tokenized = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, batched=True)

        return tokenized, text, label


        