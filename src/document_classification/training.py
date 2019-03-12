import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from document_classification.dataset import Dataset, sample
from document_classification.model import initialize_model, DocumentClassificationModel
from document_classification.load import load_data
from document_classification.preprocess import preprocess_data
from document_classification.split import split_data
from document_classification.utils import set_seeds, collate_fn, TensorboardLogger, model_summary

# Logger
ml_logger = logging.getLogger("ml_logger")

def training_setup(config):
    """Set up the training configuration."""
    # Set seeds
    set_seeds(seed=config["seed"], cuda=config["cuda"])

    # Expand file paths
    config["vectorizer_file"] = os.path.join(config["experiment_dir"], config["vectorizer_file"])
    config["model_file"] = os.path.join(config["experiment_dir"], config["model_file"])

    # Check CUDA
    if not torch.cuda.is_available():
        config["device"] = False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")

    return config


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


class Trainer(object):
    def __init__(self, experiment_id, dataset, model, model_file, device, shuffle,
               num_epochs, batch_size, learning_rate, early_stopping_criteria):
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode="min", factor=0.5, patience=1)
        self.train_state = {
            "done_training": False,
            "stopped_early": False,
            "early_stopping_step": 0,
            "early_stopping_best_val": 1e8,
            "early_stopping_criteria": early_stopping_criteria,
            "learning_rate": learning_rate,
            "epoch_index": 0,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": -1,
            "test_accuracy": -1,
            "model_filename": model_file}
        self.tensorboard = TensorboardLogger(log_dir="tensorboard/{0}".format(self.experiment_id))

    def update_train_state(self):
        # Verbose
        ml_logger.info("Epoch: {0}/{1} | lr: {2:.2E} | train_loss: {3:.3f} | train_accuracy: {4:.1f}% | val_loss: {5:.3f} | val_accuracy: {6:.1f}%".format(
            self.train_state["epoch_index"]+1, self.num_epochs, self.train_state["learning_rate"],
            self.train_state["train_loss"][-1], self.train_state["train_accuracy"][-1],
            self.train_state["val_loss"][-1], self.train_state["val_accuracy"][-1]))

        # Tensorboard log scalar metrics
        self.tensorboard.scalar_summary("lr", self.train_state["learning_rate"], self.train_state["epoch_index"]+1)
        for metric in ["train_loss", "val_loss"]:
            value = self.train_state[metric][-1]
            self.tensorboard.scalar_summary("loss/{0}".format(metric), value, self.train_state["epoch_index"]+1)
        for metric in ["train_accuracy", "val_accuracy"]:
            value = self.train_state[metric][-1]
            self.tensorboard.scalar_summary("accuracy/{0}".format(metric), value, self.train_state["epoch_index"]+1)

        # Tensorboard log historgram weights
        for param, value in self.model.named_parameters():
            param = param.replace(".", "/")
            self.tensorboard.histo_summary(param, value.data.cpu().numpy(), self.train_state["epoch_index"]+1)
            try:
                self.tensorboard.histo_summary(param+"/grad", value.grad.data.cpu().numpy(), self.train_state["epoch_index"]+1)
            except AttributeError as e:
                continue

        # Save one model at least
        if self.train_state["epoch_index"] == 0:
            torch.save(self.model.state_dict(), self.train_state["model_filename"])
            self.train_state["stopped_early"] = False

        # Save model if performance improved
        elif self.train_state["epoch_index"] >= 1:
            loss_tm1, loss_t = self.train_state["val_loss"][-2:]

            # If loss worsened
            if loss_t >= self.train_state["early_stopping_best_val"]:
                # Update step
                self.train_state["early_stopping_step"] += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state["early_stopping_best_val"]:
                    torch.save(self.model.state_dict(), self.train_state["model_filename"])

                # Reset early stopping step
                self.train_state["early_stopping_step"] = 0

            # Stop early ?
            self.train_state["stopped_early"] = self.train_state["early_stopping_step"] \
              >= self.train_state["early_stopping_criteria"]
        return self.train_state

    def run_train_loop(self):

        ml_logger.info("==> Training:")

        for epoch_index in range(self.num_epochs):
            self.train_state["epoch_index"] = epoch_index

            # Iterate over train dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            self.dataset.set_split("train")
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.0
            running_accuracy = 0.0
            self.model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                self.optimizer.zero_grad()

                # step 2. compute the output
                y_pred = self.model(batch_dict["X"])

                # step 3. compute the loss
                loss = self.loss_func(y_pred, batch_dict["y"])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                self.optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                accuracy_t = compute_accuracy(y_pred, batch_dict["y"])
                running_accuracy += (accuracy_t - running_accuracy) / (batch_index + 1)

            self.train_state["train_loss"].append(running_loss)
            self.train_state["train_accuracy"].append(running_accuracy)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            self.dataset.set_split("val")
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.
            running_accuracy = 0.
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred = self.model(batch_dict["X"])

                # step 3. compute the loss
                loss = self.loss_func(y_pred, batch_dict["y"])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                accuracy_t = compute_accuracy(y_pred, batch_dict["y"])
                running_accuracy += (accuracy_t - running_accuracy) / (batch_index + 1)

            self.train_state["val_loss"].append(running_loss)
            self.train_state["val_accuracy"].append(running_accuracy)

            self.train_state = self.update_train_state()
            self.scheduler.step(self.train_state["val_loss"][-1])
            if self.train_state["stopped_early"]:
                break

    def run_test_loop(self):
        self.dataset.set_split("test")
        batch_generator = self.dataset.generate_batches(
            batch_size=self.batch_size, collate_fn=collate_fn,
            shuffle=self.shuffle, device=self.device)
        running_loss = 0.0
        running_accuracy = 0.0
        self.model.eval()

        true = []
        pred = []
        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred =  self.model(batch_dict["X"])
            y_prob, indices = y_pred.max(dim=1)
            indices_list = indices.cpu().numpy().tolist()

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict["y"])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            accuracy_t = compute_accuracy(y_pred, batch_dict["y"])
            running_accuracy += (accuracy_t - running_accuracy) / (batch_index + 1)

            # Store
            true.extend(batch_dict["y"].cpu())
            pred.extend(indices_list)

        self.train_state["test_loss"] = running_loss
        self.train_state["test_accuracy"] = running_accuracy

        # Logging
        ml_logger.info("==> Test performance:\nTest loss: {0:.3f}\nTest Accuracy: {1:.1f}%".format(
            self.train_state["test_loss"], self.train_state["test_accuracy"]))

        # Metrics
        metrics = precision_recall_fscore_support(true, pred)

        # Results
        performance = {}
        classes = list(self.dataset.vectorizer.y_vocab.token_to_idx.keys())
        for i in range(len(classes)):
            _class = classes[i]
            precision = metrics[0][i]
            recall = metrics[1][i]
            f1 = metrics[2][i]
            num_samples = np.float64(metrics[3][i])
            performance[_class] = {"precision": precision, "recall": recall,
                               "f1": f1, "num_samples": num_samples}
        ml_logger.info("==> Performance:\n{0}".format(
            json.dumps(performance, indent=4, sort_keys=True)))

        return performance

    def save_train_state(self, experiment_dir):
        self.train_state["done_training"] = True
        ml_logger.info("==> Training complete!")
        train_state_filepath = os.path.join(experiment_dir, "train_state.json")
        with open(train_state_filepath, "w") as fp:
            json.dump(self.train_state, fp)


def training_operations(config):
    """Training operations."""
    # Load data
    df = load_data(data_file=config["data_file"])

    # Split data
    split_df = split_data(
        df=df, shuffle=config["shuffle"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"])

    # Preprocessing
    preprocessed_df = preprocess_data(split_df)

    # Load dataset and vectorizer
    dataset = Dataset.load_dataset_and_make_vectorizer(preprocessed_df)
    dataset.save_vectorizer(config["vectorizer_file"])
    vectorizer = dataset.vectorizer
    sample(dataset=dataset)

    # Initializing model
    model = initialize_model(config=config, vectorizer=vectorizer)

    # Model summary
    inputs = torch.zeros((1, 18), dtype=torch.long) # [length, batch_size]
    ml_logger.info("\n".join([line for line in model_summary(model, inputs)]))

    # Training
    trainer = Trainer(
        experiment_id=config["experiment_id"], dataset=dataset, model=model,
        model_file=config["model_file"], device=config["device"],
        shuffle=config["shuffle"], num_epochs=config["num_epochs"],
        batch_size=config["batch_size"], learning_rate=config["learning_rate"],
        early_stopping_criteria=config["early_stopping_criteria"])
    trainer.run_train_loop()

    # Testing
    trainer.train_state["performance"] = trainer.run_test_loop()

    # Save all results
    trainer.save_train_state(config["experiment_dir"])

