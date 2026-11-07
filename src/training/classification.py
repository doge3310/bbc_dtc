from torch.utils.data import DataLoader

from src.config import *
from src.training.utils import Learn
from src.model.transformer import Transformer


def learn(
    dataloader: DataLoader,
    device: str = "cpu"
) -> Transformer:

    model = Transformer(
        seq_len=SEQ_LENTH,
        d_model=D_MODEL,
        vocab_size=DICT_SIZE,
        n_head=ATTENTION_HEAD,
        num_layers=N_LAYERS,
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

    for epoch in range(EPOCH):
        for src, tgt in dataloader:
            loss, out = learn_cycle.step(
                src=src,
                tgt=tgt
            )
            print(loss, tgt)

        print(epoch, loss, tgt, out)

    return model
