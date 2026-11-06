"""Initialize dataset from .csv source"""
import csv
from torch.utils.data import Dataset
from torch import tensor
from pathlib import Path

from src.preferences import SEQ_LENTH
from src.interfaces.idata import ITokenizer


class BBCDataset(Dataset):
    def __init__(self, dataset_dir: str, tokenizer: ITokenizer, tgt_len: int):
        self.dataset = []
        self.statistic = {}
        self.label_stats = {}
        self.tokenizer = tokenizer
        self.tgt_len = tgt_len

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

        self.dictionary = self.tokenizer.forward(" ".join((item[0] for item in self.dataset)))
        self._tokenize()

    def _tokenize(self):
        labels = sorted(list(self.label_stats.keys()))

        for index, data in enumerate(self.dataset):
            self.dataset[index][0] = self.tokenizer.tokenize(data[0])[: self.tgt_len]
            self.dataset[index][1] = labels.index(data[1])

    def text_sumary(self):
        """Print data about raw text (not tokenized)"""
        print(self.statistic)
        print(self.label_stats)

    def dataset_summary(self):
        pass
    
    def export_dictionary_as_file(self, directory: Path) -> None:
        with open(directory / "dictionary.csv", mode="w", encoding="utf-8", newline="") as file:
            dict_file = csv.writer(file)

            dict_file.writerow(["word", "token"])
            dict_file.writerows(self.dictionary.items())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return [tensor(i) for i in self.dataset[index]]
