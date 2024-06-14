import os
import sys
import keras
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras.models import Sequential
from keras.datasets import mnist
from keras.preprocessing import image
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
sys.path.append('../')

save_dir = 'predictions'

def train_model_with_convolution_neural_networks():
    print('Training model using convolution neural networks')
    epochs = 12
    batch_size = 128
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    checkpoint_path = "training/cp-{epoch:04d}.weights.h5"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    os.listdir(checkpoint_dir)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[cp_callback],)
    score = model.evaluate(x_test, y_test, verbose=0)
    keras.saving.save_model(model, 'convolution_neural_network_model.keras')

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


def setup_model():
    model = load_model('convolution_neural_network_model.keras')
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict(model, file):
    img = load_and_preprocess_image(file)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    print('Predicted class:', predicted_class)


def use_trained_model():
    model = setup_model()
    predict(model, './my-numbers/number-0.png')
    predict(model, './my-numbers/number-1.png')
    predict(model, './my-numbers/number-2.png')
    predict(model, './my-numbers/number-3.png')
