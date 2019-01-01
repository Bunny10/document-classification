import os
import json
import sys
import logging
from logging.handlers import RotatingFileHandler

def create_dirs(dirpath):
    """Creating directories.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def setup_logger(name, log_file, level=logging.DEBUG):
    """Function setup as many loggers as you want.
    """
    # Level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format
    _format = '%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s'
    formatter = logging.Formatter(_format)

    # Stream
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    # File rotating
    handler = RotatingFileHandler(
        log_file, maxBytes=1024*1024*1, backupCount=1)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger