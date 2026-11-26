"""Initialize dataset from .csv source"""
import csv
import logging
from torch.utils.data import Dataset
from torch import tensor
from pathlib import Path

from src.interfaces.idata import ITokenizer


logger = logging.getLogger(__name__)


def import_data(
    dataset_dir: Path
) -> tuple[list, dict]:

    dataset = []
    labels = {}

    with open(dataset_dir, "r", encoding="utf-8") as data:
        data = csv.reader(data)

        for row_num, row in enumerate(data):
            if row_num == 0:
                continue

            labels[row[1]] = labels.get(row[1], 0) + 1

            dataset.append(row)

    return dataset, labels


class BBCDataset(Dataset):
    def __init__(self,
        rows: list[list[str]],
        tokenizer: ITokenizer,
        tgt_len: int,
        labels: dict[str, int]
) -> None:

        self.tokenizer = tokenizer
        self.tgt_len = tgt_len
        self.labels = labels
        self.dataset: list = rows.copy()
        self.rows_num = len(self.dataset)

        logger.info("dataset loaded: rows: %d, labels: %s",
            self.rows_num,
            list(labels.keys())
        )

        self._tokenize()

    def _tokenize(self):
        labels = sorted(list(self.labels.keys()))

        for index, data in enumerate(self.dataset):
            self.dataset[index][0] = self.tokenizer.tokenize(data[0])[: self.tgt_len]
            self.dataset[index][1] = labels.index(data[1])

    def __len__(self):
        return self.rows_num

    def __getitem__(self, index):
        return [tensor(i) for i in self.dataset[index]]
