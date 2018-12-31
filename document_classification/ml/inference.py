import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate

from document_classification.config import BASE_DIR
from document_classification.ml.preprocess import preprocess_data
from document_classification.ml.vectorizer import Vectorizer
from document_classification.ml.dataset import InferenceDataset, sample
from document_classification.ml.model import initialize_model

class Inference(object):
    def __init__(self, model, vectorizer, device="cpu"):
        self.model = model.to(device)
        self.vectorizer = vectorizer
        self.device = device

    def predict(self, dataset):
        # Batch generator
        batch_generator = dataset.generate_batches(
            batch_size=len(dataset), collate_fn=default_collate,
            shuffle=False, device=self.device)
        self.model.eval()

        # Predict
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict['X'], apply_softmax=True)

            # Top k nationalities
            y_prob, indices = torch.topk(y_pred, k=len(self.vectorizer.y_vocab))
            probabilities = y_prob.detach().to('cpu').numpy()[0]
            indices = indices.detach().to('cpu').numpy()[0]

            results = []
            for probability, index in zip(probabilities, indices):
                y = self.vectorizer.y_vocab.lookup_index(index)
                results.append({'y': y, 'probability': np.float64(probability)})

        return results


def inference_operations(experiment_id, X):
    """Inference operations.
    """

    # Load train config
    config_filepath = os.path.join(
        BASE_DIR, "experiments", experiment_id, "config.json")
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)

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

    # Create inference dataset
    infer_df = pd.DataFrame([X], columns=['X'])
    infer_df = preprocess_data(df=infer_df)
    infer_dataset = InferenceDataset(df=infer_df, vectorizer=vectorizer)

    # Predict
    results = inference.predict(dataset=infer_dataset)

    return results
