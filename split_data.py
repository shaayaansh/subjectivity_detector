import pandas as pd 
import numpy as np 
import pickle
import os
from sklearn.model_selection import train_test_split

data_path = "Data/MPQA/MPQA.pkl"

print(data_path)

with open(data_path, "rb") as f:
    data = pickle.load(f)


print(len(data))
print("==============")
print(data[0])