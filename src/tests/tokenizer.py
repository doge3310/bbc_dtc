import torch
import unittest

from src.config import *
from src.data_init.tokenizer import WordTokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer(
            pad_token=PAD,
            unk_token=UNK,
            dict_size=DICT_SIZE
        )
        self.text = "Hello! My name is Pipisi. This is test"
        self.dct = self.tokenizer.forward(self.text)

    def test_vocab(self):
        self.assertLessEqual(len(self.dct), 10)

    def test_tokenize(self):
        src = "hello, unk_word"
        out = self.tokenizer.tokenize(src)
        
        self.assertEqual(len(out), 2)
        self.assertNotEqual(out[0], UNK)
        self.assertEqual(out[1], UNK)


if __name__ == "__main__":
    unittest.main()
