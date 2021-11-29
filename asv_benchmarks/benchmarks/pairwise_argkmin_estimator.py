import numpy as np
from threadpoolctl import ThreadpoolController

from .common import Benchmark

from sklearn.metrics import (
    pairwise_distances_argmin,
    pairwise_distances_argmin_min
)

from sklearn.cluster import (
    AffinityPropagation,
    Birch,
    MeanShift,
    SpectralClustering,
)

from sklearn.neighbors import NearestNeighbors

from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    TSNE,
)

from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
)

class PairwiseDistancesArgKminBenchmark(Benchmark):

    param_names = [
        "n_threads",
        "n_train",
        "n_test",
        "n_features",
    ]

    params = [
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1000, 10_000, 100_000],
        [1000, 10_000, 100_000],
        [50, 100],
    ]

    def __init__(self):
        self.controller = ThreadpoolController()
        self.controlled_apis = None

    def setup(self, n_threads, n_train, n_test, n_features):
        rng = np.random.RandomState(0)
        self.n_threads = n_threads
        self.X_train = rng.rand(n_train, n_features)
        self.X_test = rng.rand(n_test, n_features)
        self.y_train = rng.randint(low=-1, high=1, size=(n_train,))

    def time_pairwise_distances_argmin(self, n_threads, n_train, n_test, n_features):
        with self.controller.limit(user_api=self.controlled_apis, limits=self.n_threads):
            pairwise_distances_argmin(X=self.X_test, Y=self.X_train)

    def time_nearest_neighbors(self, n_threads, n_train, n_test, n_features):
        with self.controller.limit(user_api=self.controlled_apis, limits=self.n_threads):
            est = NearestNeighbors(n_neighbors=10).fit(X=self.X_train)
            est.kneighbors(self.X_test)

    def time_pairwise_distances_argmin_min(self, n_threads, n_train, n_test, n_features):
        with self.controller.limit(user_api=self.controlled_apis, limits=self.n_threads):
            pairwise_distances_argmin_min(X=self.X_test, Y=self.X_train)

