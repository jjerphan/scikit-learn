import numpy as np

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

    param_names = ["n_train", "n_test", "n_features"]
    params = [
        [1000, 10_000, int(1e7)],
        [1000, 10_000, 100_000],
        [100],
    ]

    def setup(self, n_train, n_test, n_features):
        rng = np.random.RandomState(0)
        self.X_train = rng.rand(n_train, n_features).astype(np.float32)
        self.X_test = rng.rand(n_test, n_features).astype(np.float32)
        self.y_train = rng.randint(low=-1, high=1, size=(n_train,))

    def time_nearest_neighbors(self, n_train, n_test, n_features):
        est = NearestNeighbors(n_neighbors=10).fit(X=self.X_train)
        est.kneighbors(self.X_test)
