import numpy as np
import torch
import pandas as pd
from config import MODEL_NAME, MAX_LENGTH
from typing import NoReturn
from utils import preprocess_title
from torch.utils.data import Dataset
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


class ShopeeNLPSiameseDataset(Dataset):
    """
    Dataset of similarity pairs with labels
    """

    def __init__(self, dataset: pd.DataFrame) -> NoReturn:
        """
        Args:
            dataset (str): pandas dataframe

        Returns:
            NoReturn
        """

        self.data = dataset[["title_1", "title_2", "label"]]

    def __len__(self):
        """Method for getting dataset length

        Returns:
            int: the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.array, np.array, int]:
        """Method for getting item from a dataset

        Args:
            index (int): index of the element of dataset

        Returns:
            Tuple[np.array, np.array, int]:
            ecoded titles, attention masks and similarity
        """

        row = self.data.iloc[index]

        titles = [
            preprocess_title(row["title_1"]),
            preprocess_title(row["title_2"])
        ]
        target = row["label"]

        text = TOKENIZER(
            titles,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_idxs = text["input_ids"]
        attention_masks = text["attention_mask"]

        return {
            "first_title_idxs": input_idxs[0],
            "second_title_idxs": input_idxs[1],
            "first_title_mask": attention_masks[0],
            "second_title_mask": attention_masks[1],
            "target": torch.tensor(target, dtype=torch.float),
        }
