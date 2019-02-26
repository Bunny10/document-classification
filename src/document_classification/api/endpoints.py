import os
from datetime import datetime
from flask import Blueprint, jsonify, make_response, request
from http import HTTPStatus
import json

from document_classification.config import ml_logger
from document_classification.api.utils import train, infer, performance, \
    get_classes, get_valid_experiment_ids, experiment_info, delete_experiment

# Define blueprint
_api = Blueprint("_api", __name__)

# Health check
@_api.route("/document-classification", methods=["GET"])
def _health_check():
    """Health check."""
    # Construct response
    response = {
        "message": HTTPStatus.OK.phrase,
        "method": request.method,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
        }

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Training
@_api.route("/document-classification/train", methods=["POST"])
def _train():
    """Training a model."""
    # Process inputs
    config_file = request.json["config_file"]

    # Training
    results = train(config_file=config_file)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Add data
    if results["status-code"] == HTTPStatus.OK:
        response["data"] = results["data"]

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Inference
@_api.route("/document-classification/infer", methods=["POST"])
@_api.route("/document-classification/infer/<experiment_id>", methods=["POST"])
def _infer(experiment_id="latest"):
    """Inference where the inputs is a pdf file."""
    # Get inputs
    X = request.json["X"]

    # Inference
    results = infer(experiment_id=experiment_id, X=X)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Add data
    if results["status-code"] == HTTPStatus.OK:
        response["data"] = results["data"]

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# List of experiments
@_api.route("/document-classification/experiments", methods=["GET"])
def _experiments():
    """Get a list of available valid experiments."""
    # Get ids
    experiment_ids = get_valid_experiment_ids()

    # Construct response
    response = {
        "data": {"experiments": experiment_ids},
        "message": HTTPStatus.OK.phrase,
        "method": request.method,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Experiement info
@_api.route("/document-classification/info", methods=["GET"])
@_api.route("/document-classification/info/<experiment_id>", methods=["GET"])
def _experiment_info(experiment_id="latest"):
    """Get experiment info."""
    # Get experiment info
    results = experiment_info(experiment_id=experiment_id)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Add data
    if results["status-code"] == HTTPStatus.OK:
        response["data"] = results["data"]

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Classes
@_api.route("/document-classification/classes", methods=["GET"])
@_api.route("/document-classification/classes/<experiment_id>", methods=["GET"])
def _classes(experiment_id="latest"):
    """Classes in an experiment."""
    # Get classes
    results = get_classes(experiment_id=experiment_id)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Add data
    if results["status-code"] == HTTPStatus.OK:
        response["data"] = results["data"]

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Performance
@_api.route("/document-classification/performance", methods=["GET"])
@_api.route("/document-classification/performance/<experiment_id>", methods=["GET"])
def _performance(experiment_id="latest"):
    """Test performance metrics across all classes."""
    # Get performance
    results = performance(experiment_id=experiment_id)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Add data
    if results["status-code"] == HTTPStatus.OK:
        response["data"] = results["data"]

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])


# Delete an experiment
@_api.route("/document-classification/delete/<experiment_id>", methods=["GET"])
def _delete(experiment_id):
    """Delete an experiment."""
    # Get ids
    results = delete_experiment(experiment_id=experiment_id)

    # Construct response
    response = {
        "message": results["message"],
        "method": request.method,
        "status-code": results["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url,
    }

    # Log
    ml_logger.info(json.dumps(response, indent=4, sort_keys=True))
    return make_response(jsonify(response), response["status-code"])
