import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F

from document_classification.utils import compute_accuracy

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

    def compile(self, learning_rate, optimizer, scheduler,
                loss_func, collate_fn, device):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.collate_fn = collate_fn
        self.device = device

    def forward_pass(self, inputs, outputs):
        # Compute the output
        y_pred = self(inputs)

        # Compute the loss
        loss = self.loss_func(y_pred, outputs)

        # compute the accuracy
        accuracy = compute_accuracy(y_pred, outputs)

        return y_pred, loss, accuracy

    def fit(self, train_dataset, val_dataset, num_epochs, batch_size):
        self.history = {
            "learning_rate": self.learning_rate,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch_index in range(num_epochs):

            # Training
            train_generator = train_dataset.generate_batches(
                batch_size=batch_size, collate_fn=self.collate_fn, device=self.device)
            running_loss = 0.0
            running_accuracy = 0.0
            self.train()

            for batch_index, batch_dict in enumerate(train_generator):

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
                running_loss += (loss.to("cpu").item() - running_loss) / (batch_index + 1)
                running_accuracy += (accuracy - running_accuracy) / (batch_index + 1)


            self.history["train_loss"].append(running_loss)
            self.history["train_accuracy"].append(running_accuracy)

            # Validation
            val_generator = val_dataset.generate_batches(
                batch_size=batch_size, collate_fn=self.collate_fn, device=self.device)
            running_loss = 0.0
            running_accuracy = 0.0
            self.eval()

            for batch_index, batch_dict in enumerate(val_generator):

                # Forward pass
                _, loss, accuracy = self.forward_pass(
                    inputs=batch_dict["X"], outputs=batch_dict["y"])

                # Update metrics
                running_loss += (loss.to("cpu").item() - running_loss) / (batch_index + 1)
                running_accuracy += (accuracy - running_accuracy) / (batch_index + 1)

            self.history["val_loss"].append(running_loss)
            self.history["val_accuracy"].append(running_accuracy)
            self.scheduler.step(self.history["val_loss"][-1])

            # Verbose
            ml_logger.info("Epoch: {0}/{1} | lr: {2:.2E} | train_loss: {3:.3f} | train_accuracy: {4:.1f}% | val_loss: {5:.3f} | val_accuracy: {6:.1f}%".format(
                epoch_index+1, num_epochs, self.history["learning_rate"],
                self.history["train_loss"][-1], self.history["train_accuracy"][-1],
                self.history["val_loss"][-1], self.history["val_accuracy"][-1]))

        return self.history

    def evaluate(self, dataset):
        generator = dataset.generate_batches(
            batch_size=min(128, len(dataset)), collate_fn=self.collate_fn, device=self.device)
        running_loss = 0.0
        running_accuracy = 0.0
        self.eval()

        true = []
        pred = []
        for batch_index, batch_dict in enumerate(generator):

            # Forward pass
            y_pred, loss, accuracy = self.forward_pass(
                inputs=batch_dict["X"], outputs=batch_dict["y"])

            # Update metrics
            running_loss += (loss.to("cpu").item() - running_loss) / (batch_index + 1)
            running_accuracy += (accuracy - running_accuracy) / (batch_index + 1)

            # Store
            y_prob, indices = y_pred.max(dim=1)
            indices_list = indices.cpu().numpy().tolist()
            true.extend(batch_dict['y'].cpu())
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

    def predict(self, X, classes):
        X = torch.LongTensor(X).unsqueeze(0)
        y_pred = self(X, apply_softmax=True)

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
        torch.save(self.state_dict(), model_filepath)




