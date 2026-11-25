"""Initialize dataloaders from dataset"""
from torch.utils.data import DataLoader

from src.data_init.dataset import BBCDataset, import_data
from src.data_init.tokenizer import WordTokenizer
from src.config import *
from src.data_init.collate import collate_fn


def split_rows(
    rows: list[list[str]],
    rations: tuple[float, float, float]
):
    pass


def construct_dataloader() -> tuple[BBCDataset, DataLoader]:
    tokenizer = WordTokenizer(
        pad_token=PAD,
        unk_token=UNK,
        dict_size=DICT_SIZE
    )
    training_dataset = BBCDataset(
        
    )
    training_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    return training_loader
