import torch
import numpy as np
from torch import Tensor
from collections import namedtuple


NodeInfo = namedtuple(
    'NodeInfo',
    [
        'label',
        'split_prop',
        'prop_dict',
    ],
)

# pylint: disable = no-member

def information_entropy(serial):
    """
    计算信息熵，以 2 为 log 的底数
    """
    flatten = serial.view(-1)  # 展平
    _, counts = torch.unique(flatten, return_counts=True)
    prob = counts / float(flatten.size(0))
    entropy = -prob.mul(prob.log2()).sum()
    return entropy

def information_entropy_with_prop(serial, labels):
    prop_unique, prop_counts = torch.unique(serial, return_counts=True)
    prop_prob = prop_counts / float( serial.size(0) )

    entropy = 0
    for item, prob in zip(prop_unique, prop_prob):
        entropy += information_entropy(labels[serial == item]) * prob

    return entropy, prop_unique


class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, Y):
        """
        """

        if not isinstance(X, Tensor):
            raise ValueError("输入必需为 ndarray.")

        if not isinstance(Y, Tensor):
            raise ValueError("输入必需为 ndarray.")

        if X.ndim != 2:
            raise ValueError("样本数据必需为二维数据")

        y_entropy = information_entropy(Y)
        idx = self.find_best_devide_property(X, Y)

        tree = self.generate_tree(X, Y)

    def find_best_devide_property(self, matrix, labels):
        """
        """
        num_sample, num_prop = matrix.shape

        tmp_entropy = float('inf')
        for idx in range(num_prop):
            entropy, prop_unique = information_entropy_with_prop(matrix[:, idx], labels)
            if entropy < tmp_entropy:
                tmp_entropy = entropy
                prop_idx = idx
                prop_items = prop_unique

        return prop_idx, prop_items

    def generate_tree(self, x, y):
        y_unique = torch.unique(y)
        num_sample, num_prop = x.shape

        if y_unique.size(0) == 1:
            node = NodeInfo(
                label = y_unique.item(),
                split_prop = None,
                prop_dict = None,
            )
            return node

        prop_idx, prop_items = self.find_best_devide_property(x, y)

        prop_dict = {}
        for item in prop_items:
            selected = x[:, prop_idx] == item
            x_next = x[selected, :][:, torch.arange(num_prop) != prop_idx]
            y_next = y[selected]
            assert x_next.size(0) == y_next.size(0)

            prop_dict[item.item()] = self.generate_tree(x_next, y_next)

        node = NodeInfo(
            label = None,
            split_prop = prop_idx,
            prop_dict = prop_dict
        )
        return node

if __name__ == "__main__":
    pass
