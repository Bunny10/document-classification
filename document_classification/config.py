import os
from flask_caching import Cache
from document_classification.utils import setup_logger, create_dirs

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Loggers
log_dir = os.path.join(BASE_DIR, 'logs'); create_dirs(log_dir)
flask_logger = setup_logger(name='werkzeug',
                            log_file=os.path.join(log_dir, 'flask.log'))
ml_logger = setup_logger(name='ml',
                         log_file=os.path.join(log_dir, 'ml.log'))

# Cache
cache = Cache(config={'CACHE_TYPE': 'simple'})

class FlaskConfig(object):
    """
    """
    # General
    SECRET_KEY = 'change-this-not-so-secret-key'
    SEND_FILE_MAX_AGE_DEFAULT = 0 # cache busting
    #PERMANENT_SESSION_LIFETIME = timedelta(minutes=60)

class DevelopmentConfig(FlaskConfig):
    """
    """
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

class ProductionConfig(FlaskConfig):
    """
    """
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 5000
