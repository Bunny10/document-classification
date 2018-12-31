import os
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








