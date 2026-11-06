import torch
from torch import nn

from src.model.transformer import Transformer
from src.data_init.dataloader import bbc_loader, dataset
from src.preferences import *
from src.training.utils import Learn, export_model
from src.data_init.dataset import BBCDataset


def export(
    model: nn.Module,
    save_dir: Path,
    dataloader: BBCDataset,
    dict_size: int = 4000,
    input_shape: tuple = (1, 200),
    device: str = "cpu",
    batch_size: int = 32,
    seq_len: int = 200
) -> None:

    export_model(
        model=model,
        save_dir=save_dir,
        dict_size=dict_size,
        device=device,
        input_shape=input_shape,
        batch_size=batch_size,
        seq_len=seq_len
    )
    
    dataloader.export_dictionary_as_file(
        directory=save_dir
    )


def learn(
    device: str = "cpu"
) -> Transformer:

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

    for epoch in range(EPOCH):
        dataset = iter(bbc_loader)

        for src, tgt in dataset:
            loss, out = learn_cycle.step(
                src=src,
                tgt=tgt
            )
            print(loss, tgt)

        print(epoch, loss, tgt, out)
    
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = learn(device=device)

    if input("save model? (y/n): ") == "y":
        export(
            model=model,
            save_dir=EXPORT_DIR,
            dataloader=dataset,
            device=device
        )


if __name__ == "__main__":
    main()    
