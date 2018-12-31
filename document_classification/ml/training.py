import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from document_classification.config import BASE_DIR, ml_logger
from document_classification.ml.utils import set_seeds, generate_unique_id, collate_fn
from document_classification.ml.load import load_data
from document_classification.ml.split import split_data
from document_classification.ml.preprocess import preprocess_data
from document_classification.ml.vocabulary import Vocabulary, SequenceVocabulary
from document_classification.ml.vectorizer import Vectorizer
from document_classification.ml.dataset import Dataset, sample
from document_classification.ml.model import DocumentClassificationModel, initialize_model

class Trainer(object):
    def __init__(self, dataset, model, model_file, save_dir, device, shuffle,
               num_epochs, batch_size, learning_rate, early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.save_dir = save_dir
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'done_training': False,
            'stopped_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': model_file}

    def update_train_state(self):

        # Verbose
        ml_logger.info("[EPOCH]: {0} | [LR]: {1} | [TRAIN LOSS]: {2:.2f} | [TRAIN ACC]: {3:.1f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.1f}%".format(
          self.train_state['epoch_index']+1, self.train_state['learning_rate'],
            self.train_state['train_loss'][-1], self.train_state['train_acc'][-1],
            self.train_state['val_loss'][-1], self.train_state['val_acc'][-1]))

        # Save one model at least
        if self.train_state['epoch_index'] == 0:
            torch.save(self.model.state_dict(), self.train_state['model_filename'])
            self.train_state['stopped_early'] = False

        # Save model if performance improved
        elif self.train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = self.train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state['early_stopping_best_val']:
                    torch.save(self.model.state_dict(), self.train_state['model_filename'])

                # Reset early stopping step
                self.train_state['early_stopping_step'] = 0

            # Stop early ?
            self.train_state['stopped_early'] = self.train_state['early_stopping_step'] \
              >= self.train_state['early_stopping_criteria']
        return self.train_state

    def compute_accuracy(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def run_train_loop(self):

        ml_logger.info("\n==> 🏋 Training:")

        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index

            # Iterate over train dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                self.optimizer.zero_grad()

                # step 2. compute the output
                y_pred = self.model(batch_dict['X'])

                # step 3. compute the loss
                loss = self.loss_func(y_pred, batch_dict['y'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                self.optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['y'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.
            running_acc = 0.
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred = self.model(batch_dict['X'])

                # step 3. compute the loss
                loss = self.loss_func(y_pred, batch_dict['y'])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['y'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)

            self.train_state = self.update_train_state()
            self.scheduler.step(self.train_state['val_loss'][-1])
            if self.train_state['stopped_early']:
                break

    def run_test_loop(self):
        self.dataset.set_split('test')
        batch_generator = self.dataset.generate_batches(
            batch_size=self.batch_size, collate_fn=collate_fn,
            shuffle=self.shuffle, device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()

        y_pred_list = []
        y_test_list = []
        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred =  self.model(batch_dict['X'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['y'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = self.compute_accuracy(y_pred, batch_dict['y'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # Store
            y_pred_list.extend(y_pred.detach())
            y_test_list.extend(batch_dict['y'].detach())

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

        # Logging
        ml_logger.info("\n==> 💯 Test performance:")
        ml_logger.info("Test loss: {0:.2f}".format(self.train_state['test_loss']))
        ml_logger.info("Test Accuracy: {0:.1f}%".format(self.train_state['test_acc']))

        return y_pred_list, y_test_list

    def plot_performance(self, show_plot=True):
        # Figure size
        plt.figure(figsize=(15,5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(self.train_state["train_loss"], label="train")
        plt.plot(self.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(self.train_state["train_acc"], label="train")
        plt.plot(self.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        # Save figure
        plt.savefig(os.path.join(self.save_dir, "performance.png"))

        # Show plots
        if show_plot:
            plt.show()

    def save_train_state(self):
        self.train_state["done_training"] = True
        ml_logger.info("==> ✅ Training complete!")
        with open(os.path.join(self.save_dir, "train_state.json"), "w") as fp:
            json.dump(self.train_state, fp)


def training_setup(config):
    """Set up training config.
    """
    # Set seeds
    set_seeds(seed=config["seed"], cuda=config["cuda"])

    # Generate experiment ID
    config["experiment_id"] = generate_unique_id()

    # Expand file paths
    config["save_dir"] = os.path.join(
        BASE_DIR, config["save_dir"], config["experiment_id"])
    os.makedirs(config["save_dir"])
    config["vectorizer_file"] = os.path.join(
        config["save_dir"], config["vectorizer_file"])
    config["model_file"] = os.path.join(
        config["save_dir"], config["model_file"])

    # Save config
    config_fp = os.path.join(config["save_dir"], "config.json")
    with open(config_fp, "w") as fp:
        json.dump(config, fp)

    # Check CUDA
    if not torch.cuda.is_available():
        config["device"] = False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")

    return config


def training_operations(config):
    """ Operations for the training procedure.
    """

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

    # Sample checks
    sample(dataset=dataset)

    # Initializing model
    model = initialize_model(config=config, vectorizer=vectorizer)

    # Training
    trainer = Trainer(
        dataset=dataset, model=model, model_file=config["model_file"],
        save_dir=config["save_dir"], device=config["device"],
        shuffle=config["shuffle"], num_epochs=config["num_epochs"],
        batch_size=config["batch_size"], learning_rate=config["learning_rate"],
        early_stopping_criteria=config["early_stopping_criteria"])
    trainer.run_train_loop()

    # Testing
    y_pred, y_test = trainer.run_test_loop()

    # Save all results
    trainer.save_train_state()

