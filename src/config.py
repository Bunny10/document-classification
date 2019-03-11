import os
import json
import logging.config

# Base directory
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# Loggers
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
log_filepath = os.path.join(CONFIGS_DIR, "logging.json")
with open(log_filepath, "r") as fp: log_config = json.load(fp)
logging.config.dictConfig(log_config)
ml_logger = logging.getLogger("ml_logger")


class FlaskConfig(object):
    """Default Flask configuration."""
    # General
    SECRET_KEY = "change-this-not-so-secret-key"
    SEND_FILE_MAX_AGE_DEFAULT = 0 # cache busting

class DevelopmentConfig(FlaskConfig):
    """Development configuration."""
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 3000

class ProductionConfig(FlaskConfig):
    """Production configuration."""
    DEBUG = False
    HOST = "0.0.0.0"
    PORT = 3000
