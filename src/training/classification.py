import logging
from torch.utils.data import DataLoader

from src.config import *
from src.training.utils import Learn
from src.model.transformer import Transformer


logger = logging.getLogger(__name__)


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
        label_smoothing=LABEL_SMOOTHING,
        device=device
    )

    for epoch in range(EPOCH):
        total_loss = 0.0

        for step, (src, tgt) in enumerate(dataloader, 1):
            loss, _ = learn_cycle.step(
                src=src,
                tgt=tgt
            )
            total_loss += loss

            logger.debug("epoch: %d, loss: %.4f",
                epoch, loss)

            if step % 50 == 0:
                logger.info("epoch: %d, step: %d, mean_loss: %.4f",
                    epoch, step, total_loss / step)

        logger.info("epoch: %d done, mean_loss: %.4f",
            epoch, total_loss / step)

    return model
