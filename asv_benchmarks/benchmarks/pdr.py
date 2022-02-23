import numpy as np

from .common import Benchmark

from sklearn.neighbors import NearestNeighbors


class PairwiseDistancesRadiusNeighborhoodBenchmark(Benchmark):

    param_names = ["n_train", "n_test", "n_features"]
    params = [
        [1000, 10_000],
        [1000, 10_000, 100_000],
        [100],
    ]

    def setup(self, n_train, n_test, n_features):
        rng = np.random.RandomState(0)
        X_train = rng.rand(n_train, n_features)
        self.X_test = rng.rand(n_test, n_features)
        self.est = NearestNeighbors().fit(X_train)

    def time_pairwise_distances_argmin(self, n_train, n_test, n_features):
        self.est.radius_neighbors(
            X=self.X_test, return_distance=True, sort_results=True
        )
