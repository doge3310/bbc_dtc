import torch
from torch import nn

from src.model.transformer import Transformer
from src.data_init.dataloader import bbc_loader
from src.preferences import *