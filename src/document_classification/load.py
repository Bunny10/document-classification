import os
import logging
import pandas as pd

# Logger
ml_logger = logging.getLogger("ml_logger")

def load_data(data_file):
    """Load the data into a Pandas DataFrame."""
    # Load into DataFrame
    df = pd.read_csv(data_file, header=0)
    df.columns = ["y", "X"]

    # Log df sample
    ml_logger.info("==> Raw data:\n{}".format(df.head()))

    return df