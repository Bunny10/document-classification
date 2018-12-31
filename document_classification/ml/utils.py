import os
import copy
import json
import numpy as np
import pandas as pd
import random
import time
from threading import Thread
import torch
import uuid

def load_config(config_filepath):
    """Load the yaml config.
    """
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)
    return config


def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def generate_unique_id():
    """Generate a unique uuid
    preceded by a epochtime.
    """
    timestamp = int(time.time())
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())

    return unique_id


def pad_seq(seq, length):
        vector = np.zeros(length, dtype=np.int64)
        vector[:len(seq)] = seq
        vector[len(seq):] = 0 # mask_index=0
        return vector


def collate_fn(batch):

    # Make a deep copy
    batch_copy = copy.deepcopy(batch)
    processed_batch = {"X": [], "y": []}

    # Get max sequence length
    max_seq_len = max([len(sample["X"]) for sample in batch_copy])

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





