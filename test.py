import torch
import os
from dataset import CustomDataset
from tqdm import tqdm
import argeparse
from torch.utils.data import ConcatDataset, DataLoader
from Model.bert_model import BertDetector
from transformers import AutoTokenizer
from sklearn.metrics import classification_report


def main(args):
    model_name = "bert-base-uncased"
    data_path = "Data"
    dataset_name = args.dataset_name
    model_load_path = os.path.join("Model", f"{dataset_name}best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model = BertDetector(model_name)
    model.load_state_dict(torch.load(model_load_path, weights_only=True))
    
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
        test_dataset = CustomDataset(dataset_path, "test", labels, tokenizer)
    
    elif dataset_name == "all":
        mpqa_dataset_path = os.path.join(data_path, "MPQA")
        news_1_dataset_path = os.path.join(data_path, "News-1")
        news_2_dataset_path = os.path.join(data_path, "News-2")

        mpqa_test_dataset = CustomDataset(mpqa_dataset_path, "test", mpqa_labels, tokenizer)
        news_1_test_dataset = CustomDataset(news_1_dataset_path, "test", news_1_labels, tokenizer)
        news_2_test_dataset = CustomDataset(news_2_dataset_path, "test", news_2_labels, tokenizer)

        # combining all the datasets
        test_dataset = ConcatDataset([mpqa_test_dataset, news_1_test_dataset, news_2_test_dataset])

    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    
    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    for batch in tqdm(test_dataloader):
        tokenized, _, labels = batch
        labels = torch.tensor(labels).to(device)
        input_ids = tokenized["input_ids"].squeeze(1).to(device)
        attention_mask = tokenized["attention_mask"].squeeze(1).to(device)

        outputs = model((input_ids, attention_mask))
        probabilities = torch.sigmoid(outputs)
        preds = torch.where(probabilities > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy().tolist())

    print("TEST PERFORMANCE: ")
    print("========================")
    print(classification_report(y_true, y_pred, target_names=["Obj", "Subj"]))