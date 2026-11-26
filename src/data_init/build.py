"""Initialize dataloaders from dataset"""
from torch.utils.data import DataLoader
import random

from src.data_init.dataset import BBCDataset, import_data
from src.data_init.tokenizer import WordTokenizer
from src.config import *
from src.data_init.collate import collate_fn


def split_rows(
    rows: list[list[str]],
    rations: tuple[float, float] = (0.8, 0.2)
) -> tuple[list, list]:

    idx = list(range(len(rows)))
    random.shuffle(idx)

    n_train = int(len(rows) * rations[0])
    train_idx = [rows[index] for index in idx[: n_train]]
    test_idx = [rows[index] for index in idx[n_train:]]

    return train_idx, test_idx


def construct_dataloader() -> tuple[DataLoader, DataLoader]:
    dataset, labels = import_data(
        dataset_dir=DATASET_DIR
    )
    tokenizer = WordTokenizer(
        pad_token=PAD,
        unk_token=UNK,
        dict_size=DICT_SIZE
    )
    train_dataset, test_dataset = split_rows(
        rows=dataset
    )

    tokenizer.forward(
        text=" ".join([item[0] for item in train_dataset])
    )

    training_dataset = BBCDataset(
        rows=train_dataset,
        tokenizer=tokenizer,
        tgt_len=SEQ_LENTH,
        labels=labels
    )
    test_dataset = BBCDataset(
        rows=test_dataset,
        tokenizer=tokenizer,
        tgt_len=SEQ_LENTH,
        labels=labels
    )

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    return training_loader, test_loader
