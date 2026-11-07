from torch import nn

from src.training.utils import Learn, export_model
from src.data_init.dataset import BBCDataset
from src.config import *


def export(
    model: nn.Module,
    save_dir: Path,
    dataset: BBCDataset,
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
    
    dataset.export_dictionary_as_file(
        directory=save_dir
    )
