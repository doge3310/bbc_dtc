import torch
from torchinfo import summary
import unittest

from src.model.transformer import Transformer
from src.config import *


class TestModel(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.model = Transformer(
            seq_len=SEQ_LENTH,
            d_model=D_MODEL,
            vocab_size=DICT_SIZE,
            n_head=ATTENTION_HEAD,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            pad_index=PAD,
            n_features=N_LABELS
        ).to(self.device)

    def test_init(self):
        with torch.no_grad():
            x = torch.randint(low=0, high=DICT_SIZE, size=[BATCH_SIZE, SEQ_LENTH]).to(self.device)
            out: torch.Tensor = self.model(x)

        self.assertEqual((BATCH_SIZE, N_LABELS), out.size())


if __name__ == "__main__":
    unittest.main()
