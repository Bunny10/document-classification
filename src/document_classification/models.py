import os
import logging
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from document_classification.dataset import Dataset
from document_classification.utils import BatchLogger, compute_accuracy, \
                                          model_summary, wrap_text, \
                                          collate_fn, class_weights

# Logger
ml_logger = logging.getLogger("ml_logger")


def initialize_model(model_config, vectorizer):
    """Initialize a model."""
    model = TextClassificationModel(
        embedding_dim=model_config["embeddings"]["embedding_dim"],
        num_embeddings=len(vectorizer.X_vocab),
        num_channels=model_config["cnn"]["num_filters"],
        filter_sizes=model_config["cnn"]["filter_sizes"],
        hidden_dim=model_config["fc"]["hidden_dim"],
        num_classes=len(vectorizer.y_vocab),
        dropout_p=model_config["fc"]["dropout_p"],
        padding_idx=vectorizer.X_vocab.mask_index,
        freeze_embeddings=model_config["embeddings"]["freeze_embeddings"])
    return model


class TextClassificationModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_channels,
                 filter_sizes, hidden_dim, num_classes, dropout_p,
                 padding_idx=0, freeze_embeddings=False):
        super(TextClassificationModel, self).__init__()

        # Emebddings
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                       num_embeddings=num_embeddings,
                                       padding_idx=padding_idx)

        # Conv weights
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_channels,
                                   kernel_size=f) for f in filter_sizes])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Freeze embeddings
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, x_in, channel_first=False, apply_softmax=False):
        """Forward pass."""

        # ╒═══════╕
        # │ Embed │
        # ╘═══════╛

        # Embed inputs
        x_emb = self.embeddings(x_in)

        # ╒════════╕
        # │ Encode │
        # ╘════════╛

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            x_emb = x_emb.transpose(1, 2)

        # Conv outputs
        z = [conv(x_emb) for conv in self.convs]
        z = [F.max_pool1d(_z, _z.size(2)) for _z in z]
        z = [_z.squeeze(2) for _z in z]

        # Concat conv outputs
        z = torch.cat(z, 1)

        # ╒════════╕
        # │ Decode │
        # ╘════════╛

        # FC layers
        z = self.dropout(z)
        z = self.fc1(z)
        y_pred = self.fc2(z)

        # Softmax
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
