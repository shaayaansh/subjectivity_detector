
def tokenize(dataset, tokenizer):
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)