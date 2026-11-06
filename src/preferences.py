"""Constants for model and dataset initialization"""
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parent
DATASET_DIR = MAIN_DIR.parent / "data" / "raw_data.csv"
EXPORT_DIR = MAIN_DIR.parent / "model_export"

BATCH_SIZE = 32
DICT_SIZE = 4000
SEQ_LENTH = 200
PAD = 0
UNK = 1
N_LABELS = 5

D_MODEL = 256
LR = 10e-5
EPOCH = 1
LABEL_SMOOTHING = 1e-5
ATTENTION_HEAD = 4
MODEL_SIZE = 8
DROPOUT = 0.1
