import torch
from typing import Dict, List
from tqdm import tqdm
from datasets.arrow_dataset import Dataset as hgDataset
from torch.utils.data import DataLoader


class TokenDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, dataset: hgDataset, tokenizer):
        input_token: Dict[str, List[int]] = {
            "input_ids": [],
            "attention_mask": [],
        }

        for description in tqdm(dataset["text"]):
            token = tokenizer(
                text=description,
                padding="max_length",  # return_tensors='pt',
                max_length=256,
                truncation=True,
            )
            input_token["input_ids"].append(token["input_ids"])
            input_token["attention_mask"].append(token["attention_mask"])

        self.encodings = input_token
        self.labels = dataset["label"]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def create_dataloader(dataset: hgDataset, tokenizer, batch_size: int):
    tokenized_dataset = TokenDataset(dataset, tokenizer)
    return DataLoader(tokenized_dataset, batch_size)
