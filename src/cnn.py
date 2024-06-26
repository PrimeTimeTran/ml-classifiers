import os
import keras
import numpy as np
from PIL import Image

from keras import optimizers
from keras import models, layers, callbacks, datasets

class CNN():
    def __init__(self):
        print('init')
        self.saved_model = 'tmp/models/cnn_model.keras'

    def train(self):
        epochs, batch_size, num_classes = 12, 128, 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=optimizers.Adadelta(),
                    metrics=['accuracy'])

        checkpoint_path = "tmp/models/cp-{epoch:04d}.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
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
        keras.saving.save_model(model, self.saved_model)

    def load_and_preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array

    def setup_model(self):
        model = models.load_model(self.saved_model)
        model.summary()
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self,model, file):
        img = self.load_and_preprocess_image(file)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print('Predicted class:', predicted_class)


    def use_trained_model(self):
        model = self.setup_model()
        self.predict(model, './my-numbers/number-0.png')
        self.predict(model, './my-numbers/number-1.png')
        self.predict(model, './my-numbers/number-2.png')
        self.predict(model, './my-numbers/number-3.png')