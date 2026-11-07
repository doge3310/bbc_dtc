"""Initialize dataloader from dataset"""
from torch.utils.data import DataLoader, Dataset

from src.data_init.dataset import BBCDataset
from src.data_init.tokenizer import WordTokenizer
from src.config import *
from src.data_init.collate import collate_fn


def construct_dataloader() -> tuple[Dataset, DataLoader]:
    tokenizer = WordTokenizer(
        pad_token=PAD,
        unk_token=UNK,
        dict_size=DICT_SIZE
    )
    dataset = BBCDataset(
        dataset_dir=DATASET_DIR,
        tokenizer=tokenizer,
        tgt_len=SEQ_LENTH
    )
    bbc_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    return dataset, bbc_loader
