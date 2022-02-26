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
        [1000, 10_000],
        [1000, 10_000, 100_000],
        [100],
    ]

    def setup(self, n_train, n_test, n_features):
        rng = np.random.RandomState(0)
        self.X_train = rng.rand(n_train, n_features).astype(np.float32)
        self.X_test = rng.rand(n_test, n_features).astype(np.float32)
        self.y_train = rng.randint(low=-1, high=1, size=(n_train,))

    def time_pairwise_distances_argmin(self, n_train, n_test, n_features):
        pairwise_distances_argmin(X=self.X_test, Y=self.X_train)

    def time_nearest_neighbors(self, n_train, n_test, n_features):
        est = NearestNeighbors(n_neighbors=10).fit(X=self.X_train)
        est.kneighbors(self.X_test)

    def time_pairwise_distances_argmin_min(self, n_train, n_test, n_features):
        pairwise_distances_argmin_min(X=self.X_test, Y=self.X_train)

    def time_affinity_propagation(self, n_train, n_test, n_features):
        est = AffinityPropagation().fit(self.X_train)
        est.predict(self.X_test)

    def time_birch(self, n_train, n_test, n_features):
        est = Birch().fit(self.X_train)
        est.predict(self.X_test)

    def time_mean_shift(self, n_train, n_test, n_features):
        est = MeanShift().fit(self.X_train)
        est.predict(self.X_test)

    def time_spectral_clustering(self, n_train, n_test, n_features):
        est = SpectralClustering().fit_predict(self.X_train)

    def time_isomap(self, n_train, n_test, n_features):
        est = Isomap().fit(self.X_train)
        est.transform(self.X_test)

    def time_locally_linear_embedding(self, n_train, n_test, n_features):
        est = LocallyLinearEmbedding().fit(self.X_train)
        est.transform(self.X_test)

    def time_tsne(self, n_train, n_test, n_features):
        est = TSNE().fit_transform(self.X_train)

    def time_label_propagation(self, n_train, n_test, n_features):
        est = LabelPropagation().fit(self.X_train, self.y_train)
        est.predict(self.X_test)

    def time_label_spreading(self, n_train, n_test, n_features):
        est = LabelSpreading().fit(self.X_train, self.y_train)
        est.predict(self.X_test)

