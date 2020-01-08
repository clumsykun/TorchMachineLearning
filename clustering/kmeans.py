"""
kmeans algorithm.
"""

# pylint: disable = no-member
# pylint: disable = not-callable

import torch
from torch import Tensor
from numpy import ndarray
from device import DEVICE


def EuclideanDistance(a, b):
    """
    a 为 M * N 矩阵，b 为 K * N 矩阵
    返回矩阵为 M * K
    注意：将样本多的矩阵放在 a 的位置
    """
    distance_list = []

    for idx in range(b.shape[0]):
        distance = torch.sqrt(
            torch.pow(a - b[idx, :], 2).sum(dim = 1)
        )
        distance_list.append(distance)

    distance_tensor = torch.stack(distance_list, dim = 1)
    return distance_tensor


class ModelKMeans(torch.nn.Module):
    def __init__(self,
                 n_clusters,
                 max_iter
                ):
        super(ModelKMeans, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.centroids  = dict(
            zip(range(self.n_clusters),
            [None]*self.n_clusters)
        )

    def forward(self, X):
        n_sample, n_features = X.shape

        # 选取前 n_clusters 行为初始中心
        self.centroids = X[:self.n_clusters, :].clone().detach()

        for i in range(self.max_iter):
            distance_matrix = EuclideanDistance(X, self.centroids)
            labels = distance_matrix.argmin(dim=1)
            
            # 更新 centroids
            for j in range(self.n_clusters):
               self.centroids[j, :] = X[labels == j, :].mean(dim=0)

        return labels

class KMeans:
    def __init__(self,
                 n_clusters=6,
                 max_iter = 1000,
                ):

        self.labels = None
        self.model = ModelKMeans(n_clusters, max_iter).to(DEVICE)

    def fit(self, X):

        if isinstance(X, ndarray):
            X = torch.from_numpy(X).to(DEVICE)
        elif isinstance(X, Tensor):
            if X.device.type == 'cpu':
                X = X.to(DEVICE)
        else:
            raise ValueError('X 的数据类型错误')

        if X.ndim != 2:
            raise ValueError('X 的维数错误')

        self.labels = self.model(X).cpu().numpy()
