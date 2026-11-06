"""Initialize dataloader from dataset"""
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from src.data_init.dataset import BBCDataset
from src.data_init.tokenizer import WordTokenizer
from src.preferences import *


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD)
    labels_stacked = torch.stack(labels)
    
    return texts_padded, labels_stacked


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
