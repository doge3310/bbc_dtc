import torch
from torch.nn.utils.rnn import pad_sequence

from src.config import PAD


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD)
    labels_stacked = torch.stack(labels)
    
    return texts_padded, labels_stacked
