import os
import json
import logging
import math
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader

from document_classification.vectorizer import Vectorizer
from document_classification.vocabulary import Vocabulary, SequenceVocabulary

# Logger
ml_logger = logging.getLogger("ml_logger")

class Dataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

    def __str__(self):
        return "<Dataset(size={0})>".format(len(self))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        X = self.vectorizer.vectorize(row.X)
        y = self.vectorizer.y_vocab.lookup_token(row.y)
        return {"X": X, "y": y}

    def get_num_batches(self, batch_size):
        return math.ceil(len(self) / batch_size)

    def generate_batches(self, batch_size, collate_fn, device):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn,
                                drop_last=False)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

