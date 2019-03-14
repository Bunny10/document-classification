import os
from collections import Counter
import json
import numpy as np
import pandas as pd
import torch

from document_classification.vocabulary import Vocabulary, SequenceVocabulary
from document_classification.utils import load_json

class Vectorizer(object):
    def __init__(self, X_vocab=None, y_vocab=None):
        self.X_vocab = X_vocab
        self.y_vocab = y_vocab

    def fit(self, df, cutoff=0):
        # Create class vocab
        self.y_vocab = Vocabulary()
        for y in sorted(set(df.y)):
            self.y_vocab.add_token(y)

        # Get word counts
        word_counts = Counter()
        for X in df.X:
            for token in X.split(" "):
                word_counts[token] += 1

        # Create X vocab
        self.X_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                self.X_vocab.add_token(word)

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

    def to_serializable(self):
        return {"X_vocab": self.X_vocab.to_serializable(),
                "y_vocab": self.y_vocab.to_serializable()}

    def load(self, vectorizer_filepath):
        contents = load_json(vectorizer_filepath)
        self.X_vocab = SequenceVocabulary.from_serializable(contents["X_vocab"])
        self.y_vocab = Vocabulary.from_serializable(contents["y_vocab"])

    def save(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.to_serializable(), fp)
