import torch
from torch import nn, optim, Tensor
from torch.export import Dim
from pathlib import Path

from src.data_init.dataset import BBCDataset


class Learn:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        pad_index: int,
        label_smoothing: float,
        device: str
    ) -> None:

        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.optim = optim.AdamW(
            params=model.parameters(),
            lr=lr,
        )
        self.model = model.to(device)

    def step(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> tuple[float, Tensor]:

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        self.optim.zero_grad()
        output = self.model(src)
        loss = self.loss_fn(output, tgt)
        loss.backward()
        self.optim.step()

        return loss.item(), output


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
