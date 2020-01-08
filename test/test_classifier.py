import torch
from classifier import DecisionTreeClassifier, NaiveBayes
import pandas as pd


def test_DecisionTreeClassifier():
    watermelon = pd.read_csv('test/watermelon.csv')
    X = torch.Tensor(watermelon.values[:, :-1])
    Y = torch.Tensor(watermelon.values[:, -1])
    dtclassifier = DecisionTreeClassifier()
    dtclassifier.fit(X, Y)


def test_NaiveBayes():
    pass


if __name__ == "__main__":
    pass
