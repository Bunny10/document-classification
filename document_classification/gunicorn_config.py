from document_classification.config import log_dir
from document_classification.utils import create_dirs
create_dirs(dirpath=log_dir)

workers = 4
bind = "0.0.0.0:5000"
loglevel = "debug"
accesslog = "logs/access.log"
errorlog = "logs/error.log"
timeout = 30
reload = True # make True for debugging
capture_output = False # make False for debugging