import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

from .common import Benchmark

from sklearn.neighbors import NearestNeighbors

# To run benchmarks defined this file, between for instance your <current_branch>
# and its fork point with upstream/main, run:
#
# asv continuous -b PairwiseDistancesReductions -e `git merge-base --fork-point upstream/main <your_branch>` <your_branch> # noqa


class NearestNeighborsBenchmark(Benchmark):

    param_names = ["n_train", "n_test", "n_features", "metric", "strategy", "dtype"]
    params = [
        [1000, 10_000, 100_000],
        [1000, 10_000, 100_000],
        [100],
        ["manhattan", "euclidean"],
        ["auto"],
        [np.float32, np.float64],
    ]

    def setup(self, n_train, n_test, n_features, metric, strategy, dtype):
        rng = np.random.RandomState(0)
        X_train = rng.rand(n_train, n_features).astype(dtype)
        X_test = rng.rand(n_test, n_features).astype(dtype)
        self.y_train = rng.randint(low=-1, high=1, size=(n_train,))
        self.metric = metric
        self.strategy = strategy

        self.k = 10

        # Motive: radius has to be scaled with the number of feature
        # Defining it as the 0.001-quantile allows to have in expectation
        # a constant amount of neighbors, regardless of the value of n_features.
        self.radius = np.quantile(
            a=cdist(X_train[:10000], X_test[:10]).ravel(), q=0.001
        )

        self.X_train = csr_matrix(X_train)
        self.X_test = csr_matrix(X_test)

        self.nn = NearestNeighbors(
            n_neighbors=self.k,
            radius=self.radius,
            algorithm='brute',
        ).fit(self.X_train)

    def time_kneighbors(
        self, n_train, n_test, n_features, metric, strategy, dtype,
    ):
        self.nn.kneighbors(self.X_test)

