import os
from collections import Counter
import json
import logging
import numpy as np
import pandas as pd
import torch

from document_classification.vocabulary import Vocabulary, SequenceVocabulary
from document_classification.utils import load_json, wrap_text

# Logger
ml_logger = logging.getLogger("ml_logger")

class Vectorizer(object):
    def __init__(self, X_vocab=None, y_vocab=None):
        self.X_vocab = X_vocab
        self.y_vocab = y_vocab

    def __str__(self):
        return "<Vectorizer(X_vocab={0}, y_vocab={1})>".format(
            len(self.X_vocab), len(self.y_vocab))

    def fit(self, df, min_token_frequency=0):
        # Create class vocab
        self.y_vocab = Vocabulary()
        for y in sorted(set(df.y)):
            self.y_vocab.add_token(y)

        # Get token counts
        tokne_counts = Counter()
        for X in df.X:
            for token in X.split(" "):
                tokne_counts[token] += 1

        # Create X vocab
        self.X_vocab = SequenceVocabulary()
        for token, token_count in tokne_counts.items():
            if token_count >= min_token_frequency:
                self.X_vocab.add_token(token)

    def vectorize(self, X):
        indices = [self.X_vocab.lookup_token(token) for token in X.split(" ")]
        indices = [self.X_vocab.begin_seq_index] + indices + \
            [self.X_vocab.end_seq_index]

        # Create vector
        X_length = len(indices)
        vector = np.zeros(X_length, dtype=np.int64)
        vector[:len(indices)] = indices

        return vector

    def unvectorize(self, vector):
        tokens = [self.X_vocab.lookup_index(index) for index in vector]
        X = " ".join(token for token in tokens)
        return X

    def vectorize_df(self, df):
        df.X = df.X.apply(self.vectorize)
        df.y = df.y.apply(self.y_vocab.lookup_token)
        return df

    def to_serializable(self):
        return {"X_vocab": self.X_vocab.to_serializable(),
                "y_vocab": self.y_vocab.to_serializable()}

    @classmethod
    def load(cls, vectorizer_filepath):
        contents = load_json(vectorizer_filepath)
        X_vocab = SequenceVocabulary.from_serializable(contents["X_vocab"])
        y_vocab = Vocabulary.from_serializable(contents["y_vocab"])
        return cls(X_vocab=X_vocab, y_vocab=y_vocab)

    def save(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.to_serializable(), fp, indent=4)
