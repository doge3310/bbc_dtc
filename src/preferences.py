"""Constants for model and dataset initialization"""
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parent
DATASET_DIR = MAIN_DIR.parent / "data" / "raw_data.csv"

BATCH_SIZE = 32
DICT_SIZE = 140
