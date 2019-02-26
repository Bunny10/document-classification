import os
import logging.config

from document_classification.utils import create_dirs, load_json

# Base directory
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../datasets")
CONFIGS_DIR = os.path.join(BASE_DIR, "../configs")
LOGS_DIR = os.path.join(BASE_DIR, "../logs")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "../experiments")

# Loggers
create_dirs(LOGS_DIR)
log_config = load_json(filepath=os.path.join(CONFIGS_DIR, "logging.json"))
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
