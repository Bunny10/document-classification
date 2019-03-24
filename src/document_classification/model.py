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

class DocumentClassificationModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_channels,
                 filter_sizes, hidden_dim, num_classes, dropout_p,
                 padding_idx=0, freeze_embeddings=False):
        super(DocumentClassificationModel, self).__init__()

        # Emebddings
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                       num_embeddings=num_embeddings,
                                       padding_idx=padding_idx)

        # Conv weights
        self.conv = nn.ModuleList([nn.Conv1d(embedding_dim, num_channels,
                                  kernel_size=f) for f in filter_sizes])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Freeze embeddings
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, X, channel_first=False, apply_softmax=False):
        """Forward pass."""

        # ╒═══════╕
        # │ Embed │
        # ╘═══════╛

        # Embed inputs
        X = self.embeddings(X)

        # ╒════════╕
        # │ Encode │
        # ╘════════╛

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            X = X.transpose(1, 2)

        # Conv outputs
        z1 = self.conv[0](X)
        z1 = F.max_pool1d(z1, z1.size(2)).squeeze(2)
        z2 = self.conv[1](X)
        z2 = F.max_pool1d(z2, z2.size(2)).squeeze(2)
        z3 = self.conv[2](X)
        z3 = F.max_pool1d(z3, z3.size(2)).squeeze(2)

        # Concat conv outputs
        z = torch.cat([z1, z2, z3], 1)

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


class Model(object):
    def __init__(self, model_config, vectorizer, model_filepath=None, tensorboard=None):
        """Initialize the model."""
        self._model = DocumentClassificationModel(
            embedding_dim=model_config["embeddings"]["embedding_dim"],
            num_embeddings=len(vectorizer.X_vocab),
            num_channels=model_config["cnn"]["num_filters"],
            filter_sizes=model_config["cnn"]["filter_sizes"],
            hidden_dim=model_config["fc"]["hidden_dim"],
            num_classes=len(vectorizer.y_vocab),
            dropout_p=model_config["fc"]["dropout_p"],
            padding_idx=vectorizer.X_vocab.mask_index,
            freeze_embeddings=model_config["embeddings"]["freeze_embeddings"])
        self.vectorizer = vectorizer
        self.model_filepath = model_filepath
        self.tensorboard = tensorboard

    def summary(self, df):
        # Model summary
        dataset = Dataset(df=pd.DataFrame(df, index=[0]),
                          vectorizer=self.vectorizer)
        loader = dataset.generate_batches(batch_size=1,
                                          collate_fn=collate_fn,
                                          device=torch.device("cpu"))
        for batch_dict in loader:
            inputs = batch_dict["X"]
            model_summary(self._model, inputs)

    def compile(self, train_df, learning_rate, early_stopping_criteria):
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=1)
        self.loss_func = nn.CrossEntropyLoss(class_weights(train_df, self.vectorizer))
        self.early_stopping_criteria = early_stopping_criteria
        self.num_bad_epochs = 0

    def process_batch(self, batch_index, batch_dict, mode):
        """Process a batch."""

        # Forward pass
        predictions = self._model(batch_dict["X"])
        batch_loss = self.loss_func(predictions, batch_dict["y"])
        batch_accuracy = compute_accuracy(predictions, batch_dict["y"])

        if mode == "train":
            # Use loss to produce gradients and update weights
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return predictions, batch_loss, batch_accuracy

    def process_epoch(self, dataset, batch_size, mode, batch_logger):
        """Process an epoch."""

        # Generate batches
        loader = dataset.generate_batches(batch_size=batch_size,
                                          collate_fn=collate_fn,
                                          device=self.device)
        start = time.time()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        if mode == "train": self._model.train()
        else: self._model.eval()

        # Process batches
        y_true = []
        y_pred = []
        for batch_index, batch_dict in enumerate(loader):

            # Process batch
            predictions, batch_loss, batch_accuracy = self.process_batch(
                batch_index=batch_index, batch_dict=batch_dict, mode=mode)

            # Store rseults
            probabilities, indices = predictions.max(dim=1)
            indices_list = indices.cpu().numpy().tolist()
            y_true.extend(batch_dict["y"].cpu())
            y_pred.extend(indices_list)

            # Update metrics
            epoch_loss += (batch_loss.to("cpu").item() - epoch_loss) / (batch_index + 1)
            epoch_accuracy += (batch_accuracy - epoch_accuracy) / (batch_index + 1)

            # Log
            batch_logger.log(batch_index=batch_index,
                             lr=self.optimizer.param_groups[0]["lr"],
                             loss=epoch_loss,
                             accuracy=epoch_accuracy,
                             start=start,
                             mode=mode)
            start = time.time()

        return epoch_loss, epoch_accuracy, y_true, y_pred

    def fit(self, train_df, val_df, test_df, num_epochs, batch_size, cuda):
        """Fit a model to a dataset."""

        # Set device
        self.device = torch.device("cuda" if cuda else "cpu")
        self._model.to(self.device)

        # Datasets
        train_dataset = Dataset(df=train_df, vectorizer=self.vectorizer)
        val_dataset = Dataset(df=val_df, vectorizer=self.vectorizer)
        test_dataset = Dataset(df=test_df, vectorizer=self.vectorizer)

        # Metrics
        self.results = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_performance": {}
        }

        # Batch loggers
        self.batch_logger = BatchLogger(train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        test_dataset=test_dataset,
                                        batch_size=batch_size)

        # Epochs
        wrap_text("Training")
        for epoch_index in range(num_epochs):
            print ("\nEpoch {0}/{1}".format(epoch_index+1, num_epochs,))

            # ╒══════════╕
            # │ Training │
            # ╘══════════╛
            train_loss, train_accuracy, _, _ = self.process_epoch(dataset=train_dataset,
                                                                  batch_size=batch_size,
                                                                  batch_logger=self.batch_logger,
                                                                  mode="train")
            self.results["train_loss"].append(train_loss)
            self.results["train_accuracy"].append(train_accuracy)

            # ╒════════════╕
            # │ Validation │
            # ╘════════════╛
            val_loss, val_accuracy, _, _ = self.process_epoch(dataset=val_dataset,
                                                              batch_size=batch_size,
                                                              batch_logger=self.batch_logger,
                                                              mode="val")
            self.results["val_loss"].append(val_loss)
            self.results["val_accuracy"].append(val_accuracy)

            # Update learning rate
            self.scheduler.step(self.results["val_loss"][-1])
            if (self.end_training()):
                break # end training

            # Log to tensorboard
            if self.tensorboard:
                self.tensorboard.log(model=self._model,
                                     results=self.results,
                                     learning_rate=self.optimizer.param_groups[0]["lr"],
                                     step=epoch_index)

        # ╒═════════╕
        # │ Testing │
        # ╘═════════╛
        ml_logger.info("\n")
        wrap_text("Testing")
        test_loss, test_accuracy, y_true, y_pred = self.process_epoch(dataset=test_dataset,
                                                                      batch_size=batch_size,
                                                                      batch_logger=self.batch_logger,
                                                                      mode="test")
        self.results["test_performance"]["test_loss"] = test_loss
        self.results["test_performance"]["test_accuracy"]= test_accuracy

        # Metrics
        metrics = precision_recall_fscore_support(y_true, y_pred)

        # Per-class performance
        self.results["test_performance"]["class_performance"] = {}
        classes = list(self.vectorizer.y_vocab.token_to_idx.keys())
        for i in range(len(classes)):
            self.results["test_performance"]["class_performance"][classes[i]] = {
                "precision": metrics[0][i]*100.0,
                "recall": metrics[1][i]*100.0,
                "f1": metrics[2][i]*100.0,
                "num_samples": np.float64(metrics[3][i])
            }
        ml_logger.info("\n")
        ml_logger.info(json.dumps(self.results["test_performance"], indent=4, sort_keys=True))

        return self.results

    def end_training(self):
        """End the training loop if number of regressing
        epochs reached the early stopping criteria."""
        save_model = True
        if self.scheduler.num_bad_epochs >= self.scheduler.patience:
            self.num_bad_epochs += 1
            save_model = False
        if self.num_bad_epochs >= self.early_stopping_criteria:
            print ("\nEnding training early!")
            return True
        else:
            if save_model:
                self.save(self.model_filepath)
            return False

    def predict(self, X):
        self._model.eval()
        self._model = self._model.to("cpu")

        # Forward pass
        X = torch.LongTensor(X).unsqueeze(0)
        y_pred = self._model(X, apply_softmax=True)
        classes = self.vectorizer.y_vocab

        # Top k nationalities
        y_prob, indices = torch.topk(y_pred, k=len(classes))
        probabilities = y_prob.detach().to("cpu").numpy()[0]
        indices = indices.detach().to("cpu").numpy()[0]

        prediction = []
        for probability, index in zip(probabilities, indices):
            y = classes.lookup_index(index)
            prediction.append({"y": y, "probability": np.float64(probability)})

        return prediction

    def save(self, model_filepath):
        torch.save(self._model.state_dict(), model_filepath)

    def load(self, model_filepath):
        self._model.load_state_dict(torch.load(model_filepath), strict=False)




