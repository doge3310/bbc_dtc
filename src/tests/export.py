import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.model.transformer import Transformer
from src.data_init.build import construct_dataloader
from src.export.onnx import export
from src.config import *


class TestExport(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory(dir=".", suffix="export_test_")
        cls.addClassCleanup(cls.temp_dir.cleanup)
        temp_path = Path(cls.temp_dir.name)
        cls.file_path = temp_path / "model.onnx"

        device = "cpu"
        model = Transformer(
            seq_len=SEQ_LENTH,
            d_model=D_MODEL,
            vocab_size=DICT_SIZE,
            n_head=ATTENTION_HEAD,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            pad_index=PAD,
            n_features=N_LABELS
        ).to(device)
        dataset, _ = construct_dataloader()

        export(
            model=model,
            save_dir=temp_path,
            dataset=dataset
        )

    def test_file_exist(self):
        self.assertTrue(self.file_path.is_file())

    def test_file_not_empty(self):
        size = self.file_path.stat().st_size
        self.assertGreater(size, 0)

if __name__ == "__main__":
    unittest.main()
