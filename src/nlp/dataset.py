import numpy as np
import pandas as pd
import config
from typing import NoReturn
from utils import preprocess_title
from torch.utils.data import Dataset
from transformers import BertTokenizer

MODEL_NAME = config.MODEL_NAME
MAX_LEN = config.MAX_LENGTH
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)


class ShopeeNLPDataset(Dataset):
    """
    Dataset of preprocessed products titles
    """

    def __init__(self, path_to_file: str) -> NoReturn:
        """
        Args:
            path_to_file (str): path to the dataset .csv file

        Returns:
            NoReturn
        """

        self.data = pd.read_csv(path_to_file)["title"].values

    def __len__(self) -> int:
        """Method for getting dataset length

        Returns:
            int: the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.array, np.array]:
        """Method for getting item from a dataset

        Args:
            index (int): index of the element of dataset

        Returns:
            Tuple[np.array, np.array]: tokenized title and It's attention mask
        """

        sentence = self.data[index]
        sentence = preprocess_title(sentence)
        text = TOKENIZER(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        return input_ids, attention_mask
