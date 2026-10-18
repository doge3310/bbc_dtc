"""Constants for model and dataset initialization"""
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parent
DATASET_DIR = MAIN_DIR.parent / "data" / "raw_data.csv"

BATCH_SIZE = 32
DICT_SIZE = 700
SEQ_LENTH = 1500
PAD = 0
UNK = 1
N_LABELS = 5

D_MODEL = 128
LR = 10e-5
EPOCH = 30
LABEL_SMOOTHING = 1e-5
ATTENTION_HEAD = 2
MODEL_SIZE = 4
DROPOUT = 0.1
