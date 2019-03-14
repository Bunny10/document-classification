import os
from datetime import datetime
from http import HTTPStatus
import json
import logging
import numpy as np
import pandas as pd
import shutil
from threading import Thread
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uuid

from config import CONFIGS_DIR, DATA_DIR, EXPERIMENTS_DIR, TENSORBOARD_DIR
from document_classification.dataset import Dataset
from document_classification.model import Model
from document_classification.utils import class_weights, clean_text, \
                                          collate_fn, load_json, \
                                          set_seeds, TensorboardLogger, \
                                          train_val_test_split
from document_classification.vectorizer import Vectorizer

# Logger
ml_logger = logging.getLogger("ml_logger")

def train(config_file):
    """Asynchronously train a model."""
    # Get config
    config = set_up(config_file)

    # Save config
    config_fp = os.path.join(config["experiment_dir"], "config.json")
    with open(config_fp, "w") as fp:
        json.dump({k: config[k] for k in set(list(config.keys())) - set(["device"])}, fp)

    # Asynchronous call
    thread = Thread(target=training_operations, args=(config,))
    thread.start()

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "experiment_id": config["experiment_id"],
            "experiment_dir": config["experiment_dir"],
        }
    }

    return results


def set_up(config_file):
    # Load config
    config_filepath = os.path.join(CONFIGS_DIR, config_file)
    config = load_json(filepath=config_filepath)

    # Set seeds
    set_seeds(seed=config["seed"], cuda=config["cuda"])

    # Create experiment dir
    config["experiment_id"] = generate_unique_id()
    config["experiment_dir"] = os.path.join(EXPERIMENTS_DIR, config["experiment_id"])
    os.makedirs(config["experiment_dir"])

    # Expand file paths
    config["data_filepath"] = os.path.join(DATA_DIR, config["data_file"])
    config["vectorizer_filepath"] = os.path.join(config["experiment_dir"], config["vectorizer_file"])
    config["model_filepath"] = os.path.join(config["experiment_dir"], config["model_file"])
    config["tensorboard_dir"] = os.path.join(TENSORBOARD_DIR, config["experiment_id"])

    # Check CUDA
    if not torch.cuda.is_available():
        config["device"] = False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")

    return config


def training_operations(config):
    """Training operations."""

    # Load data
    df = pd.read_csv(config["data_filepath"], header=0)
    df.columns = ["X", "y"]

    # Preprocess data
    df.X = df.X.apply(clean_text)
    ml_logger.info(df.head(1))

    # Split data
    train_df, val_df, test_df = train_val_test_split(
        df=df, shuffle=True, min_samples_per_class=5,
        train_size=0.7, val_size=0.15, test_size=0.15)
    ml_logger.info("train: {0}, val: {1}, test: {2}".format(
        len(train_df), len(val_df), len(test_df)))

    # Vectorizer
    vectorizer = Vectorizer()
    vectorizer.fit(df=train_df, cutoff=0)

    # Datasets
    train_dataset = Dataset(df=train_df, vectorizer=vectorizer)
    val_dataset = Dataset(df=val_df, vectorizer=vectorizer)
    test_dataset = Dataset(df=test_df, vectorizer=vectorizer)

    # Model
    tensorboard = TensorboardLogger(log_dir=config["tensorboard_dir"])
    model = Model(config=config, vectorizer=vectorizer, tensorboard=tensorboard)
    ml_logger.info("==> Initialized model:\n{0}".format(model._model.named_modules))

    # Compile
    learning_rate = config["learning_rate"]
    optimizer = optim.Adam(model._model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    loss_func = nn.CrossEntropyLoss(class_weights(train_df, vectorizer))
    model.compile(learning_rate=learning_rate,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  loss_func=loss_func,
                  collate_fn=collate_fn,
                  device=config["device"])

    # Train
    history = model.fit(train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        num_epochs=config["num_epochs"],
                        batch_size=config["batch_size"])

    # Evaluate
    history["test_loss"], history["test_accuracy"], history["performance"] = \
        model.evaluate(test_dataset)

    # Save model and vectorizer
    model.save(config["model_filepath"])
    vectorizer.save(config["vectorizer_filepath"])

    # Save history
    with open(os.path.join(config["experiment_dir"], "history.json"), "w") as fp:
        json.dump(history, fp)
    ml_logger.info("==> History:\n{0}".format(
        json.dumps(history, indent=4, sort_keys=True)))


def predict(experiment_id, X):
    """Inference for an input."""
    # Validate experiment id
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Inference operations
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    config = load_json(config_filepath)

    # Load vectorizer
    vectorizer = Vectorizer()
    vectorizer.load(config["vectorizer_filepath"])

    # Load trained model
    model = Model(config=config, vectorizer=vectorizer)
    model.load(config["model_filepath"])

    # Predict
    prediction = model.predict(vectorizer.vectorize(clean_text(X)))

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "prediction": prediction,
        }
    }

    return results


def generate_unique_id():
    """Generate a unique uuid preceded by a epochtime."""
    timestamp = datetime.now().isoformat()
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())
    return unique_id


def get_experiment_info(experiment_id):
    """ Get training info for the experiment."""
    # Define experiment info filepaths
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    history_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "history.json")

    # Load files
    config = load_json(filepath=config_filepath)
    history = load_json(filepath=history_filepath)

    # Join info
    experiment_info = {**config, **history}

    return experiment_info


def get_experiment_ids():
    """Get list of valid experiments."""
    # Get experiements
    experiment_ids = [f for f in os.listdir(EXPERIMENTS_DIR) \
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, f))]

    return sorted(experiment_ids)


def validate_experiment_id(experiment_id):
    """Validate the experiment id."""
    # Get available experiment ids
    experiment_ids = get_experiment_ids()

    # Latest experiment_id
    if experiment_id == "latest":
        experiment_id = experiment_ids[-1]

    # Check experiment id is valid
    if experiment_id not in experiment_ids:
        raise ValueError("Experiment id {0} is not valid.".format(experiment_id))

    return experiment_id


def experiment_info(experiment_id):
    """Get experiment info."""
    # Validate experiment id
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Get experiment info
    experiment_info = get_experiment_info(experiment_id=experiment_id)

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "experiment_info": experiment_info
        }
    }

    return results


def get_classes(experiment_id):
    """Classes for an experiment."""
    # Validate experiment id
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Load config
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    config = load_json(filepath=config_filepath)

    # Get classes
    vectorizer = Vectorizer()
    vectorizer.load(config["vectorizer_filepath"])
    classes = list(vectorizer.y_vocab.token_to_idx.keys())

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "classes": classes
        }
    }

    return results


def delete_experiment(experiment_id):
    """Delete an experiment."""
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Delete the experiment dir
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    shutil.rmtree(experiment_dir)

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

    return results
