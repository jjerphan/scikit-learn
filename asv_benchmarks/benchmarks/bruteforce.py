import numpy as np

from scipy import sparse

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_argmin
from sklearn.manifold import Isomap
from sklearn.cluster import Birch, DBSCAN, OPTICS

from .common import Benchmark


class BruteForceNearestNeighborsBenchmark(Benchmark):
    """
    Benchmarks for KNeighborsMixin.kneighbors and
    when using algorithm='brute'.
    """

    param_names = ["n_train", "n_test", "n_features", "dtype"]
    params = (
        [100_000],
        [100_000],
        [50, 100, 500],
        [np.float64, np.float32],
    )

    def setup(self, *params):
        n_train, n_test, n_features, dtype = params
        self.nn = NearestNeighbors(
            n_neighbors=10,
            algorithm="brute",
        )

        rng = np.random.RandomState(0)

        self.X_train = sparse.rand(
            n_train, n_features, density=.05, format="csr", dtype=dtype, random_state=rng
        )
        self.X_test = sparse.rand(
            n_test, n_features, density=.05, format="csr", dtype=dtype, random_state=rng
        )

        self.nn.fit(X=self.X_train)

    def time_kneighbors(self, *params):
        self.nn.kneighbors(self.X_test, return_distance=True)

