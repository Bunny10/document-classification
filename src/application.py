import os
from flask import Flask, jsonify, make_response, request
from http import HTTPStatus

from config import DevelopmentConfig, ProductionConfig
from api.endpoints import _api

# Define Flask app
application = Flask(__name__)
application.url_map.strict_slashes = False

# Choose config
application.config.from_object(DevelopmentConfig)

# Register blueprints
application.register_blueprint(_api)

# BAD_REQUEST
@application.errorhandler(400)
def bad_request(error):
    """Redirect all bad requests."""
    response = {
        "message": HTTPStatus.BAD_REQUEST.phrase,
        "method": request.method,
        "status-code": HTTPStatus.BAD_REQUEST,
        "url": request.url,
        }
    return make_response(jsonify(response), response["status-code"])

# NOT_FOUND
@application.errorhandler(404)
def not_found(error):
    """Redirect all nonexistent URLS."""
    response = {
        "message": HTTPStatus.NOT_FOUND.phrase,
        "method": request.method,
        "status-code": HTTPStatus.NOT_FOUND,
        "url": request.url,
        }
    return make_response(jsonify(response), response["status-code"])

# INTERNAL_SERVER_ERROR
@application.errorhandler(500)
def internal_server_error(error):
    """Internal server error."""
    response = {
        "message": HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
        "method": request.method,
        "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
        "url": request.url,
        }
    return make_response(jsonify(response), response["status-code"])