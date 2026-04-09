#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------
# Load token ids (memmap)
# ----------------------------------------------
bin_path = Path("../data_cleaned/wiki_train_ids.bin")
assert bin_path.exists(), f"Missing. {bin_path}, please generate"

data = np.memmap(bin_path, dtype=np.unit16, mode="r")
N = data.shape[0]
print(f"Total tokens: {N}")

# ----------------------------------------------
# Hyper parameters
# ----------------------------------------------
vocab_size = 30_000
block_size = 128
batch_size = 32
embedding_dimensions = 128
n_head = 8
n_layer = 4
max_steps = 2_000
eval_interval = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(3674)

# ----------------------------------------------
#
# ----------------------------------------------

