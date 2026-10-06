"""Initialize dataloader from dataset"""
from torch.utils.data import DataLoader

from src.data_init.dataset import BBCDataset
from src.data_init.tokenizer import BPETokenizer
from src.preferences import *


tokenizer = BPETokenizer(
    max_tokens_count=DICT_SIZE,
    unk_id=UNK,
    pad_id=PAD
)
dataset = BBCDataset(
    dataset_dir=DATASET_DIR,
    tokenizer=tokenizer,
    tgt_len=SEQ_LENTH
)
bbc_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

if __name__ == "__main__":
    bbc_loader = iter(bbc_loader)
    data = next(bbc_loader)

    print(data[0][0])
    print([i.size() for i in data])
