import torch
from torch import nn

from src.model.transformer import Transformer
from src.data_init.dataloader import bbc_loader
from src.preferences import *
from src.training.utils import Learn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        seq_len=SEQ_LENTH,
        d_model=D_MODEL,
        vocab_size=DICT_SIZE,
        n_head=ATTENTION_HEAD,
        num_layers=MODEL_SIZE,
        dropout=DROPOUT,
        pad_index=PAD,
        n_features=N_LABELS
    )
    learn_cycle = Learn(
        model=model,
        lr=LR,
        pad_index = PAD,
        label_smoothing=LABEL_SMOOTHING,
        device=device
    )
    dataset = iter(bbc_loader)

    for epoch in range(EPOCH):
        for src, tgt in dataset:
            loss = learn_cycle.step(
                src=src,
                tgt=tgt
            )
            print(loss)

        print(epoch, loss)


if __name__ == "__main__":
    main()
