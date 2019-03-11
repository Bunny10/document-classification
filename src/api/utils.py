import os
from datetime import datetime
from http import HTTPStatus
import json
import logging
import shutil
from threading import Thread
import time
import uuid

from config import CONFIGS_DIR, DATA_DIR, EXPERIMENTS_DIR
from document_classification.dataset import Dataset
from document_classification.inference import inference_operations
from document_classification.training import training_operations, training_setup
from document_classification.utils import load_json

# Logger
ml_logger = logging.getLogger("ml_logger")

def train(config_file):
    """Asynchronously train a model."""
    # Load config
    config_filepath = os.path.join(CONFIGS_DIR, config_file)
    config = load_json(filepath=config_filepath)

    # Generate unique experiment ID
    config["experiment_id"] = generate_unique_id()

    # Define paths
    config["data_file"] = os.path.join(DATA_DIR, config["data_file"])
    config["experiment_dir"] = os.path.join(EXPERIMENTS_DIR, config["experiment_id"])
    os.makedirs(config["experiment_dir"])

    # Training set up
    config = training_setup(config=config)

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


def predict(experiment_id, X):
    """Inference for an input."""
    # Validate experiment id
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Inference operations
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    results = inference_operations(config_filepath, X=X)

    return results


def generate_unique_id():
    """Generate a unique uuid preceded by a epochtime."""
    timestamp = int(time.time())
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())
    return unique_id


def get_experiment_info(experiment_id):
    """ Get training info for the experiment."""
    # Define experiment info filepaths
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    train_state_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "train_state.json")

    # Load files
    config = load_json(filepath=config_filepath)
    train_state = load_json(filepath=train_state_filepath)

    # Join info
    experiment_info = {**config, **train_state}

    return experiment_info


def get_valid_experiment_ids():
    """Get list of valid experiments."""
    # Get experiements
    experiment_ids = [f for f in os.listdir(EXPERIMENTS_DIR) \
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, f))]

    # Only show valid experiments
    valid_experiment_ids = []
    for experiment_id in experiment_ids:
        experiment_details = get_experiment_info(experiment_id)
        if experiment_details["done_training"]:
            valid_experiment_ids.append(experiment_id)

    # Sort
    valid_experiment_ids = sorted(valid_experiment_ids)

    return valid_experiment_ids


def validate_experiment_id(experiment_id):
    """Validate the experiment id."""
    # Get available experiment ids
    experiment_ids = get_valid_experiment_ids()

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
    vectorizer = Dataset.load_vectorizer_only(
        vectorizer_filepath=config["vectorizer_file"])
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


def get_performance(experiment_id):
    """Test performance from all classes for a trained model."""
    # Validate experiment id
    try:
        experiment_id = validate_experiment_id(experiment_id)
    except ValueError as e:
        return {"message": str(e), "status-code": HTTPStatus.INTERNAL_SERVER_ERROR}

    # Load train state
    train_state_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "train_state.json")
    train_state = load_json(filepath=train_state_filepath)

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "performance": train_state["performance"]
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
