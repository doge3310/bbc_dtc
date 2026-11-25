from torch import nn
import torch
from pathlib import Path
import logging
from torch.export import Dim

from src.data_init.dataset import BBCDataset
from src.config import *


logger = logging.getLogger(__name__)


def export_model(
    model: nn.Module,
    save_dir: Path,
    dict_size: int,
    input_shape: tuple,
    device: str,
    batch_size: int,
    seq_len: int
) -> None:

    model.eval()
    input_ids = torch.randint(0, dict_size, input_shape).to(device)
    batch = Dim("batch", min=1, max=batch_size)
    seq_lenght = Dim("seq_len", min=1, max=seq_len)
    
    dynamic_shapes = {
        "x": {0: batch, 1: seq_lenght}
    }
    
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=model,
        f=str(save_dir / "model.onnx"),
        args=input_ids,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamic_shapes=dynamic_shapes,
        dynamo=True
    )


def export(
    model: nn.Module,
    save_dir: Path,
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

    # dataset.export_dictionary_as_file(
    #     directory=save_dir
    # )

    logger.info("dictionary exported to %s", save_dir / "dictionary.csv")
