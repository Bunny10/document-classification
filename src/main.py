import os
from argparse import ArgumentParser
from datetime import datetime
from http import HTTPStatus
import json
import logging
import numpy as np
import pandas as pd
import shutil
from threading import Thread
import time
import uuid

from config import DATA_DIR, EXPERIMENTS_DIR, TENSORBOARD_DIR
from document_classification.preprocessor import Preprocessor
from document_classification.dataset import Dataset
from document_classification.models import initialize_model
from document_classification.model import Model
from document_classification.utils import class_weights, \
                                          collate_fn, load_json, load_yaml, wrap_text, \
                                          set_seeds, TensorboardLogger, \
                                          train_val_test_split, \
                                          load_data, save_yaml
from document_classification.vectorizer import Vectorizer

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--data_csv", dest="data_csv", required=True,
                    help="location of data csv")
parser.add_argument("--model_configuration_file", dest="model_configuration_file", required=True,
                    help="don't print status messages to stdout")
args = parser.parse_args()

# Read config
config = load_yaml(filepath=args.model_configuration_file)

### REMOVE BELOW ###
config["filepaths"] = {}
config["filepaths"]["model_filepath"] = "exp1/model.pth"
config["filepaths"]["results_filepath"] = "exp1/results.json"
config["filepaths"]["vectorizer_filepath"] = "exp1/vectorizer.json"
config["filepaths"]["preprocessor_filepath"] = "exp1/preprocessor.json"
config["filepaths"]["model_configuration_filepath"] = "exp1/model_configuration.yaml"
### REMOVE ABOVE ###

save_yaml(obj=config,
          filepath=config["filepaths"]["model_configuration_filepath"])

# Load data
df = load_data(data_csv=args.data_csv)
column_names = {
    config['data']['input_feature']: "X",
    config['data']['output_feature']: "y",
}
df = df.rename(columns=column_names)

# Preprocess data
preprocessor = Preprocessor(lower=config["preprocessing"]["lower"],
                            char_level=config["preprocessing"]["char_level"],
                            filters=config["preprocessing"]["filters"])
df = preprocessor.clean_df(df)

# Split data
train_df, val_df, test_df = train_val_test_split(df=df,
                                                 train_size=config["splitting"]["train_size"],
                                                 val_size=config["splitting"]["val_size"],
                                                 test_size=config["splitting"]["test_size"],
                                                 min_samples_per_class=config["splitting"]["min_samples_per_class"],
                                                 shuffle=config["splitting"]["shuffle"])

# Vectorizer
vectorizer = Vectorizer()
vectorizer.fit(df=train_df,
               min_token_frequency=config["preprocessing"]["min_token_frequency"])

# Model
tensorboard = TensorboardLogger(log_dir="tensorboard/experiment")
model = Model(build_fn=initialize_model,
              model_config=config["model"],
              vectorizer=vectorizer,
              model_filepath=config["filepaths"]["model_filepath"],
              tensorboard=tensorboard)
model.summary(df=train_df)

# Compile
model.compile(train_df=train_df,
              learning_rate=config["training"]["learning_rate"],
              early_stopping_criteria=config["training"]["early_stopping_criteria"])

# Train
results = model.fit(train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    num_epochs=config["training"]["num_epochs"],
                    batch_size=config["training"]["batch_size"],
                    cuda=config["training"]["cuda"])

# Save
preprocessor.save(filepath=config["filepaths"]["preprocessor_filepath"])
vectorizer.save(filepath=config["filepaths"]["vectorizer_filepath"])
with open(config["filepaths"]["results_filepath"], "w") as fp:
    json.dump(model.results, fp, indent=4)

# ╒════════════╕
# │ Prediction │
# ╘════════════╛

# Load preprocessor
preprocessor = Preprocessor.load(filepath=config["filepaths"]["preprocessor_filepath"])

# Load vectorizer
vectorizer = Vectorizer.load(filepath=config["filepaths"]["vectorizer_filepath"])

# Load trained model
model = Model(build_fn=initialize_model,
              model_config=config["model"],
              vectorizer=vectorizer)
model.load(config["filepaths"]["model_filepath"])

# Predict
X = "Global warming is inevitables, scientists warn."
prediction = model.predict(vectorizer.vectorize(preprocessor.clean(X)))
print (json.dumps(prediction, indent=4))

