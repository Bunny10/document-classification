import os
import pandas as pd

from document_classification.config import ml_logger

def load_data(data_file):
    """Load the data into a Pandas DataFrame."""
    # Load into DataFrame
    df = pd.read_csv(data_file, header=0)
    df.columns = ["y", "X"]

    # Log df sample
    ml_logger.info("==> Raw data:\n{}".format(df.head()))

    return df