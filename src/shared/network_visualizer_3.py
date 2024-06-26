import numpy as np
from kviz.dense import DenseGraph

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import sklearn.datasets as datasets


if __name__ == "__main__":
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy")

    dg = DenseGraph(model)
    dg.render()


# if __name__ == "__main__":

#     model = Sequential()
#     model.add(Dense(3, input_shape=(2,), activation='relu'))
#     model.add(Dense(1, activation='relu'))
#     model.compile(loss="binary_crossentropy")

#     centers = [[.5, .5]]
#     t, _ = datasets.make_blobs(n_samples=50, centers=centers, cluster_std=.1)
#     X = np.array(t)
#     Y = np.array([1 if x[0] - x[1] >= 0 else 0 for x in X])

#     history = model.fit(X, Y)

#     # Initialize DenseGraph after fitting the model
#     dg = DenseGraph(model)
#     dg.animate_activations(X, duration=.3)