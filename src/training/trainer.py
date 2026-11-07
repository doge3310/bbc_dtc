import torch
from torch import nn

from src.data_init.build import construct_dataloader
from src.config import *
from src.export.onnx import export
from src.training.classification import learn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, dataloader = construct_dataloader()
    model = learn(
        device=device,
        dataloader=dataloader)

    export(
        model=model,
        save_dir=EXPORT_DIR,
        dataset=dataset,
        device=device
    )


if __name__ == "__main__":
    main()
