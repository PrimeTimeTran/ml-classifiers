from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        model = Sequential()
        model.add(Conv2D(20, 5, 5, border_mode="same", input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
        model.add(Conv2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
        model.add(Conv2D(100, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model
