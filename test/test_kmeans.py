import matplotlib.pyplot as plt
from clustering.kmeans import KMeans
from sklearn.datasets.samples_generator import make_blobs
from torch import Tensor

# ssh 问题
plt.switch_backend('agg')


def test_kmeans():
    X, y = make_blobs(
        n_samples=1000,
        centers=[
            [-1, -1, 4],
            [0, 1, 5],
            [3, 4, 12],
            [6, 8, 10],
            [-3, 7, 13],
        ],
        cluster_std=0.2,
        n_features=2,
    )

    model = KMeans(n_clusters=5, max_iter=10000)
    model.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=model.labels)
    plt.savefig('out/kmeans.png')

    print('actual labels: ', y)
    print('kmeans labels: ', model.labels)


if __name__ == "__main__":
    pass
