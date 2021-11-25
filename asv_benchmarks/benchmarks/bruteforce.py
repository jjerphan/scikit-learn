import numpy as np

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_argmin
from sklearn.manifold import Isomap
from sklearn.cluster import Birch, DBSCAN, OPTICS

from .common import Benchmark

N_SAMPLES = [10_000]
N_FEATURES = [50, 100, 500]


class BruteForceNearestNeighborsBenchmark(Benchmark):
    """
    Benchmarks for KNeighborsMixin.kneighbors and
    RadiusNeighborsMixin.radius_neighbors,
    when using algorithm='brute'.
    """

    param_names = ["n_train", "n_test", "n_features", "k_radius"]
    params = (
        N_SAMPLES,
        N_SAMPLES,
        N_FEATURES,
        [(1, 1), (10, 10), (100, 100), (1000, 1000)],
    )

    def setup(self, *params):
        n_train, n_test, n_features, (k, radius) = params
        self.nn = NearestNeighbors(
            n_neighbors=k,
            radius=radius * np.log(n_features),
            algorithm="brute",
        )

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))
        self.X_test = self.rng.random_sample((n_test, n_features))

        self.nn.fit(X=self.X_train)

    def time_kneighbors(self, *params):
        self.nn.kneighbors(self.X_test, return_distance=True)


class BruteForcePairwiseDistancesArgminBenchmark(Benchmark):
    """
    Benchmarks for KNeighborsMixin.kneighbors and
    RadiusNeighborsMixin.radius_neighbors,
    when using algorithm='brute'.
    """

    param_names = ["n_train", "n_test", "n_features"]
    params = (
        N_SAMPLES,
        N_SAMPLES,
        N_FEATURES,
    )

    def setup(self, *params):
        n_train, n_test, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))
        self.X_test = self.rng.random_sample((n_test, n_features))

    def time_pairwise_distances_argmin(self, *params):
        pairwise_distances_argmin(
                X=self.X_test,
                Y=self.X_train,
        )

    def time_pairwise_distances_argmin_min(self, *params):
        pairwise_distances_argmin_min(
                X=self.X_test,
                Y=self.X_train,
        )


class BruteForceBirchBenchmark(Benchmark):
    """
    Benchmarks for Birch.
    """

    param_names = ["n_train", "n_test", "n_features"]
    params = (
        N_SAMPLES,
        N_SAMPLES,
        N_FEATURES,
    )

    def setup(self, *params):
        n_train, n_test, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))
        self.X_test = self.rng.random_sample((n_test, n_features))

        self.est_fit = Birch()
        self.est_predict = Birch()
        self.est_predict.fit(X=self.X_train)

    # def time_fit(self, *params):
    #     self.est_fit.fit(self.X_train)

    def time_predict(self, *params):
        self.est_predict.predict(self.X_test)


class BruteForceOPTICSBenchmark(Benchmark):
    """
    Benchmarks for OPTICS.
    """

    param_names = ["n_train", "n_features"]
    params = (
        N_SAMPLES,
        [10, 50, 100, 500],
    )

    def setup(self, *params):
        n_train, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))

        self.est_fit = OPTICS(algorithm="brute")

    # def time_fit(self, *params):
    #     self.est_fit.fit(self.X_train)


class BruteForceDBSCANBenchmark(Benchmark):
    """
    Benchmarks for DBSCAN.
    """

    param_names = ["n_train", "n_features"]
    params = (
        N_SAMPLES,
        N_FEATURES,
    )

    def setup(self, *params):
        n_train, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))

        self.est_fit = DBSCAN(algorithm="brute")

    # def time_fit(self, *params):
    #     self.est_fit.fit(self.X_train)


class BruteForceIsomapBenchmark(Benchmark):
    """
    Benchmarks for Isomap.
    """

    param_names = ["n_train", "n_test", "n_features"]
    params = (
        N_SAMPLES,
        N_SAMPLES,
        N_FEATURES,
    )

    def setup(self, *params):
        n_train, n_test, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))
        self.X_test = self.rng.random_sample((n_test, n_features))

        self.est_fit = Isomap(neighbors_algorithm="brute")
        self.est_transform = Isomap(neighbors_algorithm="brute")
        self.est_transform.fit(X=self.X_train)

    # def time_fit(self, *params):
    #     self.est_fit.fit(self.X_train)

    def time_transform(self, *params):
        self.est_transform.transform(self.X_test)


class BruteForceLocalOutlierFactorBenchmark(Benchmark):
    """
    Benchmarks for LocalOutlierFactor.
    """

    param_names = ["n_train", "n_features"]
    params = (N_SAMPLES, N_FEATURES)

    def setup(self, *params):
        n_train, n_features = params

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))

        self.est_fit = LocalOutlierFactor(algorithm="brute")

    # def time_fit(self, *params):
    #     self.est_fit.fit(self.X_train)
