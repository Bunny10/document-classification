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

from document_classification.utils import BatchLogger, compute_accuracy, model_summary, box

# Logger
ml_logger = logging.getLogger("ml_logger")


class DocumentClassificationModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_channels,
                 hidden_dim, num_classes, dropout_p, padding_idx=0):
        super(DocumentClassificationModel, self).__init__()

        # Emebddings
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                       num_embeddings=num_embeddings,
                                       padding_idx=padding_idx)

        # Conv weights
        self.conv = nn.ModuleList([nn.Conv1d(embedding_dim, num_channels,
                                  kernel_size=f) for f in [2,3,4]])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Toggle to False to freeze embeddings
        self.embeddings.weight.requires_grad = True

    def forward(self, X, channel_first=False, apply_softmax=False):
        """Forward pass."""
        # Embed
        X = self.embeddings(X)

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

        # FC layers
        z = self.dropout(z)
        z = self.fc1(z)
        y_pred = self.fc2(z)

        # Softmax
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred


class Model(object):
    def __init__(self, config, vectorizer, tensorboard=None):
        """Initialize the model."""
        self._model = DocumentClassificationModel(
            embedding_dim=config["embedding_dim"],
            num_embeddings=len(vectorizer.X_vocab),
            num_channels=config["cnn"]["num_filters"],
            hidden_dim=config["fc"]["hidden_dim"],
            num_classes=len(vectorizer.y_vocab),
            dropout_p=config["fc"]["dropout_p"],
            padding_idx=vectorizer.X_vocab.mask_index).to(config["device"])
        self.vectorizer = vectorizer
        self.device = config["device"]
        self.tensorboard = tensorboard

        # Model summary
        inputs = torch.zeros((1, 18), dtype=torch.long)
        model_summary(self._model, inputs)

    def compile(self, learning_rate, optimizer, scheduler, loss_func, collate_fn):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.collate_fn = collate_fn

    def forward_pass(self, inputs, outputs):
        y_pred = self._model(inputs)
        loss = self.loss_func(y_pred, outputs)
        accuracy = compute_accuracy(y_pred, outputs)
        return y_pred, loss, accuracy

    def fit(self, train_dataset, val_dataset, num_epochs, batch_size, verbose=True):

        from document_classification.utils import box
        box("TRAINING")

        self.history = {
            "learning_rate": self.learning_rate,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        self.batch_logger = BatchLogger(train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        batch_size=batch_size)

        for epoch_index in range(num_epochs):

            print ("\nEpoch {0}/{1}".format(epoch_index+1, num_epochs,))

            # Training
            train_loader = train_dataset.generate_batches(
                batch_size=batch_size, collate_fn=self.collate_fn, device=self.device)
            start = time.time()
            running_train_loss = 0.0
            running_train_accuracy = 0.0
            self._model.train()

            for train_batch_index, batch_dict in enumerate(train_loader):

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                _, loss, accuracy = self.forward_pass(
                    inputs=batch_dict["X"], outputs=batch_dict["y"])

                # Use loss to produce gradients
                loss.backward()

                # Use optimizer to take gradient step
                self.optimizer.step()

                # Update metrics
                running_train_loss += (loss.to("cpu").item() - running_train_loss) / (train_batch_index + 1)
                running_train_accuracy += (accuracy - running_train_accuracy) / (train_batch_index + 1)

                # Log batch
                if verbose:
                    self.batch_logger.log(batch_index=train_batch_index,
                                          lr=self.history["learning_rate"],
                                          train_loss=running_train_loss,
                                          train_acc=running_train_accuracy,
                                          start=start)
                    start = time.time()


            self.history["train_loss"].append(running_train_loss)
            self.history["train_accuracy"].append(running_train_accuracy)

            # Validation
            val_loader = val_dataset.generate_batches(
                batch_size=batch_size, collate_fn=self.collate_fn, device=self.device)
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            self._model.eval()

            for val_batch_index, batch_dict in enumerate(val_loader):

                # Forward pass
                _, loss, accuracy = self.forward_pass(
                    inputs=batch_dict["X"], outputs=batch_dict["y"])

                # Update metrics
                running_val_loss += (loss.to("cpu").item() - running_val_loss) / (val_batch_index + 1)
                running_val_accuracy += (accuracy - running_val_accuracy) / (val_batch_index + 1)

                # Log batch
                if verbose:
                    self.batch_logger.log(batch_index=train_batch_index+val_batch_index+1,
                                          lr=self.history["learning_rate"],
                                          train_loss=running_train_loss,
                                          train_acc=running_train_accuracy,
                                          val_loss=running_val_loss,
                                          val_acc=running_val_accuracy,
                                          start=start)
                    start = time.time()

            self.history["val_loss"].append(running_val_loss)
            self.history["val_accuracy"].append(running_val_accuracy)
            self.scheduler.step(self.history["val_loss"][-1])

            # Log to tensorboard
            if self.tensorboard:
                self.tensorboard.log(model=self._model, history=self.history, step=epoch_index)

        return self.history

    def evaluate(self, dataset):
        loader = dataset.generate_batches(
            batch_size=min(128, len(dataset)), collate_fn=self.collate_fn, device=self.device)
        running_loss = 0.0
        running_accuracy = 0.0
        self._model.eval()

        true = []
        pred = []
        for batch_index, batch_dict in enumerate(loader):

            # Forward pass
            y_pred, loss, accuracy = self.forward_pass(
                inputs=batch_dict["X"], outputs=batch_dict["y"])

            # Update metrics
            running_loss += (loss.to("cpu").item() - running_loss) / (batch_index + 1)
            running_accuracy += (accuracy - running_accuracy) / (batch_index + 1)

            # Store
            y_prob, indices = y_pred.max(dim=1)
            indices_list = indices.cpu().numpy().tolist()
            true.extend(batch_dict["y"].cpu())
            pred.extend(indices_list)

        # Metrics
        metrics = precision_recall_fscore_support(true, pred)

        # Results
        performance = {}
        classes = list(dataset.vectorizer.y_vocab.token_to_idx.keys())
        for i in range(len(classes)):
            _class = classes[i]
            precision = metrics[0][i]
            recall = metrics[1][i]
            f1 = metrics[2][i]
            num_samples = np.float64(metrics[3][i])
            performance[_class] = {"precision": precision, "recall": recall,
                               "f1": f1, "num_samples": num_samples}

        return running_loss, running_accuracy, performance

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
        ml_logger.info("")
        box("Test Performance")
        ml_logger.info(json.dumps(self.history["performance"], indent=4, sort_keys=True))
        torch.save(self._model.state_dict(), model_filepath)

    def load(self, model_filepath):
        self._model.load_state_dict(torch.load(model_filepath), strict=False)




