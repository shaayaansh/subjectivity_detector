import pandas as pd 
import numpy as np 
import pickle
import os
from sklearn.model_selection import train_test_split

data_path = "Data/MPQA/MPQA.pkl"
save_file_path = "Data/MPQA"

with open(data_path, "rb") as f:
    data = pickle.load(f)

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=100)
train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=100)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
val_df = pd.DataFrame(val_data)

train_df = train_df[["sentence", "answer"]]
val_df = val_df[["sentence", "answer"]]
test_df = test_df[["sentence", "answer"]]

train_df["answer"] = train_df["answer"].apply(lambda x: 1 if x == "yes" else 0)
val_df["answer"] = val_df['answer'].apply(lambda x: 1 if x == "yes" else 0)
test_df["answer"] = test_df["answer"].apply(lambda x: 1 if x == "yes" else 0)

train_df.to_csv(os.path.join(save_file_path, "train.csv"))
val_df.to_csv(os.path.join(save_file_path, "val.csv"))
test_df.to_csv(os.path.join(save_file_path, "test.csv"))

