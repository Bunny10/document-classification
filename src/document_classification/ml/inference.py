import os
from http import HTTPStatus
import json
import numpy as np
import pandas as pd
import torch

from document_classification.config import EXPERIMENTS_DIR, ml_logger
from document_classification.utils import load_json
from document_classification.ml.vectorizer import Vectorizer
from document_classification.ml.model import initialize_model
from document_classification.ml.dataset import InferenceDataset
from document_classification.ml.preprocess import preprocess_data
from document_classification.ml.utils import collate_fn

class Inference(object):
    def __init__(self, model, vectorizer, device="cpu"):
        self.model = model.to(device)
        self.vectorizer = vectorizer
        self.device = device

    def predict(self, dataset):
        # Batch generator
        batch_generator = dataset.generate_batches(
            batch_size=len(dataset), collate_fn=collate_fn,
            shuffle=False, device=self.device)
        self.model.eval()

        # Predict
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict["X"], apply_softmax=True)

            # Top k nationalities
            y_prob, indices = torch.topk(y_pred, k=len(self.vectorizer.y_vocab))
            probabilities = y_prob.detach().to("cpu").numpy()[0]
            indices = indices.detach().to("cpu").numpy()[0]

            predictions = []
            for probability, index in zip(probabilities, indices):
                y = self.vectorizer.y_vocab.lookup_index(index)
                predictions.append({"y": y, "probability": np.float64(probability)})

        return predictions


def inference_operations(experiment_id, X):
    """Inference operations.
    """
    # Load config
    config_filepath = os.path.join(EXPERIMENTS_DIR, experiment_id, "config.json")
    config = load_json(filepath=config_filepath)

    # Load vectorizer
    with open(config["vectorizer_file"]) as fp:
        vectorizer = Vectorizer.from_serializable(json.load(fp))

    # Initializing model
    model = initialize_model(config=config, vectorizer=vectorizer)

    # Load model
    model.load_state_dict(torch.load(config["model_file"]))
    model = model.to("cpu")

    # Initialize inference
    inference = Inference(model=model, vectorizer=vectorizer)

    # Filler input
    filler_class = list(vectorizer.y_vocab.token_to_idx.keys())[0] # random filler y

    # Inference dataset
    infer_df = pd.DataFrame([[X, filler_class]], columns=["X", "y"])
    infer_df = preprocess_data(df=infer_df)
    infer_dataset = InferenceDataset(df=infer_df, vectorizer=vectorizer)

    # Predict
    predictions = inference.predict(dataset=infer_dataset)

    # Results
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "predictions": predictions,
            "X": infer_df["X"][0]
        }
    }

    return results
