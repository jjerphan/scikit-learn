import numpy as np
from scipy.sparse import csr_matrix

from .common import Benchmark

from sklearn.neighbors import NearestNeighbors

# To run benchmarks defined this file, between for instance your <current_branch>
# and its fork point with upstream/main, run:
#
# asv continuous -b NearestNeighborsBenchmark -e upstream/main HEAD


class NearestNeighborsBenchmark(Benchmark):

    # Overriding the inherited default individual timeout of 500 seconds
    timeout = 1000  # seconds

    param_names = [
        "n_train",
        "n_test",
        "n_features",
        "metric",
        "strategy",
        "use_sparse_datasets",
        "dtype",
    ]

    params = [
        [10_000, 100_000, 100_000, 1_000_000],
        [1_000, 10_000, 100_000, 100_000],
        [100],
        ["manhattan", "euclidean"],
        ["auto"],
        [False, True],
        [np.float32, np.float64],
    ]

    def setup(
        self, n_train, n_test, n_features, metric, strategy, use_sparse_datasets, dtype
    ):
        rng = np.random.RandomState(0)
        X_train = rng.rand(n_train, n_features).astype(dtype)
        X_test = rng.rand(n_test, n_features).astype(dtype)

        if use_sparse_datasets:
            # We set 70% of elements to zeros for a fair scenario
            mask_train = rng.rand(n_train, n_features) < 0.7
            mask_test = rng.rand(n_test, n_features) < 0.7

            X_train[mask_train] = 0.0
            X_test[mask_test] = 0.0

            X_train = csr_matrix(X_train)
            X_test = csr_matrix(X_test)

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = rng.randint(low=-1, high=1, size=(n_train,))
        self.metric = metric
        # Strategy is not used in this case, but we keep it in case
        # we want to benchmark different strategies for the backend.
        self.strategy = strategy

        self.k = 10

        self.nn = NearestNeighbors(
            n_neighbors=self.k, algorithm="brute", metric=self.metric
        ).fit(self.X_train)

    def time_kneighbors(
        self,
        n_train,
        n_test,
        n_features,
        metric,
        strategy,
        use_sparse_datasets,
        dtype,
    ):
        self.nn.kneighbors(self.X_test, return_distance=True)
