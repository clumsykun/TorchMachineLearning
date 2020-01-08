import torch
from torch import Tensor
from device import DEVICE


class ModelNaiveBayes(torch.nn.Module):
    def __init__(self):
        super(ModelNaiveBayes, self).__init__()

    def forward(self, X):
        pass


class NaiveBayes:
    def __init__(self):
        self.model = ModelNaiveBayes().to(DEVICE)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
