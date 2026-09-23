"""Initialize dataset from .csv source"""
import csv
from torch.utils.data import Dataset

from src.data_init.tokenizer import BPETokenizer


class BBCDataset(Dataset):
    def __init__(self, dataset_dir: str, tokenizer: BPETokenizer):
        self.dataset = []
        self.statistic = {}
        self.label_stats = {}

        with open(dataset_dir, "r", encoding="utf-8") as data:
            data = csv.reader(data)

            for row_num, row in enumerate(data):
                if row_num == 0:
                    continue

                row_lenth = len(row[0])

                self.label_stats[row[1]] = self.label_stats.get(row[1], 0) + 1

                self.statistic["min_text_len"] = min(
                    self.statistic.get("min_text_len", row_lenth),
                    row_lenth)

                self.statistic["max_text_len"] = max(
                    self.statistic.get("max_text_len", row_lenth),
                    row_lenth)

                self.dataset.append(row)

        self.statistic["rows_count"] = len(self.dataset)

        self._tokenize(
            dataset=" ".join((item[0] for item in self.dataset)),
            tokenizer=tokenizer)

    def _tokenize(self, dataset: list, tokenizer: BPETokenizer):
        tokenizer.forward(dataset)

        # return tokenizer.tokenize(dataset)

    def sumary(self):
        print(self.statistic)
        print(self.label_stats)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
