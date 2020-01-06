import torch
from decision_tree.model import information_entropy, DecisionTreeClassifier
import pandas as pd


def test_information_entropy():

    x = torch.Tensor([1, 1, 1, 1, 2])
    y = information_entropy(x)
    print(y)


def test_DecisionTreeClassifier():
    watermelon = pd.read_csv('test/watermelon.csv')
    X = torch.Tensor(watermelon.values[:, :-1])
    Y = torch.Tensor(watermelon.values[:, -1])
    dtclassifier = DecisionTreeClassifier()
    dtclassifier.fit(X, Y)


if __name__ == "__main__":
    pass