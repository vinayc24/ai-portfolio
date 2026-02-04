from datasets import load_dataset
from transformers import AutoTokenizer
# import torch

from config import MODEL_NAME, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length = MAX_LENGTH
    )

def get_datasets():
    dataset = load_dataset("ag_news")

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(20000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))    
    tokenized = dataset.map(tokenize, batched=True)

    tokenized.set_format(
        type = "torch",
        columns = ["input_ids","attention_mask", "label"]

    )
    return tokenized["train"], tokenized["test"]