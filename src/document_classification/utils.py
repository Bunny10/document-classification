import os
import copy
import json
import numpy as np
import torch

def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def pad_seq(seq, length):
    """Pad inputs to create uniformly sized inputs."""
    vector = np.zeros(length, dtype=np.int64)
    vector[:len(seq)] = seq
    vector[len(seq):] = 0 # mask_index=0
    return vector


def collate_fn(batch):
    """Custom collat function for batch processing."""
    # Make a deep copy
    batch_copy = copy.deepcopy(batch)
    processed_batch = {"X": [], "y": []}

    # Get max sequence length
    max_seq_len = max([len(sample["X"]) for sample in batch_copy])

    # CNN filter length requirement
    max_seq_len = max(4, max_seq_len)

    # Pad
    for i, sample in enumerate(batch_copy):
        seq = sample["X"]
        y = sample["y"]
        padded_seq = pad_seq(seq, max_seq_len)
        processed_batch["X"].append(padded_seq)
        processed_batch["y"].append(y)

    # Convert to appropriate tensor types
    processed_batch["X"] = torch.LongTensor(
        processed_batch["X"])
    processed_batch["y"] = torch.LongTensor(
        processed_batch["y"])

    return processed_batch
