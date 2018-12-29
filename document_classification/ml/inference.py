import os
import numpy as np
import torch

class Inference(object):
    def __init__(self, model, vectorizer, device):
        self.model = model.to(device)
        self.vectorizer = vectorizer
        self.device = device

    def predict(self, dataset):
        # Batch generator
        batch_generator = dataset.generate_batches(
            batch_size=len(dataset), shuffle=False, device=self.device)
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