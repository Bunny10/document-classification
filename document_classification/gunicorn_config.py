import os
from document_classification.config import log_dir
from document_classification.utils import create_dirs
from document_classification.application import application
create_dirs(dirpath=log_dir)

# http://docs.gunicorn.org/en/stable/settings.html
workers = 4
bind = "{0}:{1}".format(application["HOST"], application["PORT"])
loglevel = "debug"
accesslog = os.path.join(log_dir, "access.log")
errorlog = os.path.join(log_dir, "error.log")
timeout = 30
reload = application["DEBUG"] # make True for debugging
capture_output = not application["DEBUG"] # make False for debugging