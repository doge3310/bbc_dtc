"""Initialize dataloader from dataset"""
from torch.utils.data import DataLoader

from src.data_init.dataset import BBCDataset
from src.preferences import \
    (DATASET_DIR, BATCH_SIZE)


dataset = BBCDataset(
    dataset_dir=DATASET_DIR
)
bbc_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

if __name__ == "__main__":
    bbc_loader = iter(bbc_loader)
    print(next(bbc_loader)[0][0])
