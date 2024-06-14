import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances_chunked

class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', algorithm='brute', n_jobs=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.algorithm == 'brute':
            distances = pairwise_distances_chunked(X_test, self.X_train, metric=self.metric)
            results = Parallel(n_jobs=self.n_jobs)(delayed(self.predict_chunk)(chunk) for chunk in distances)
            y_pred = np.concatenate(results)
        else:
            raise ValueError("Unsupported algorithm. Please use 'brute'.")
        return y_pred

    def predict_chunk(self, distances):
        y_pred_chunk = []
        for dist in distances:
            nearest_indices = np.argsort(dist)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            y_pred_chunk.append(self.majority_vote(nearest_labels))
        return np.array(y_pred_chunk)

    def majority_vote(self, labels):
        counts = np.bincount(labels)
        return np.argmax(counts)
