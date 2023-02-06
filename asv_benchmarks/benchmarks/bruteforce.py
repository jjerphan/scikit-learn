import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark

N_SAMPLES_TRAIN = [10_000]
N_SAMPLES_TEST = [1_000]
N_FEATURES = [100]


class BruteForceKNeighborsClassifier(Benchmark):
    """Benchmarks for KNeighborsClassifier.predict when using algorithm='brute'."""

    param_names = ["n_train", "n_test", "n_features"]
    params = (
        N_SAMPLES_TRAIN,
        N_SAMPLES_TEST,
        N_FEATURES,
    )

    def setup(self, *params):
        n_train, n_test, n_features = params
        self.knc = KNeighborsClassifier(
            n_neighbors=1000, algorithm="brute", metric="manhattan"
        )

        self.rng = np.random.RandomState(0)
        self.X_train = self.rng.random_sample((n_train, n_features))
        self.X_test = self.rng.random_sample((n_test, n_features))
        self.y = self.rng.randint(low=0, high=2, size=(n_train,))

        self.knc.fit(X=self.X_train, y=self.y)

    def time_predict_proba(self, *params):
        self.knc.predict_proba(self.X_test)
