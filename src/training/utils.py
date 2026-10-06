import torch
from torch import nn, optim


class Learn:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        pad_index: int,
        label_smoothing: float,
        device: str):

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
        tgt: torch.Tensor):

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        self.optim.zero_grad()
        output = self.model(src)
        loss = self.loss_fn(output, tgt)
        loss.backward()
        self.optim.step()

        return loss.item()
