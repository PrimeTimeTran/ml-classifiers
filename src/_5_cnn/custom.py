import os
import sys
import keras
import numpy as np
from PIL import Image
from matplotlib import style
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from shared.mnist_loader import MNIST
from shared.utils import setup_save_directory, create_log_file, image_file_name, get_file_name, create_pickle

def model():
    type = 'KNN'
    setup_save_directory()
    style.use('ggplot')
    log_file = create_log_file(f'{type}-summary.log')
    sys.stdout = log_file

    print('\nLoading MNIST Data...')
    data = MNIST()

    print('\nLoading Training Data...')
    img_train, labels_train = data.load_training()
    train_img = np.array(img_train)
    train_labels = np.array(labels_train)

    print('\nLoading Testing Data...')
    img_test, labels_test = data.load_testing()
    test_img = np.array(img_test)
    test_labels = np.array(labels_test)

    x = train_img
    y = train_labels

    print('\nPreparing Classifier Training and Validation Data...')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1)

    print('\nClassifier with n_neighbors = 5, algorithm = auto, n_jobs = 10')
    print('\nPickling the Classifier for Future Use...')
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)

    clf.fit(x_train, y_train)

    clf = create_pickle(clf, type)

    print('\nCalculating Accuracy of trained Classifier...')
    confidence = clf.score(x_test, y_test)

    print('\nMaking Predictions on Validation Data...')
    y_pred = clf.predict(x_test)

    print('\nCalculating Accuracy of Predictions...')
    accuracy = accuracy_score(y_test, y_pred)

    print('\nCreating Confusion Matrix...')
    conf_mat = confusion_matrix(y_test, y_pred)

    print('\nTrained Classifier Confidence: ', confidence)
    print('\nPredicted Values: ', y_pred)
    print('\nAccuracy of Classifier on Validation Image Data: ', accuracy)
    print('\nConfusion Matrix: \n', conf_mat)

    plt.matshow(conf_mat)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(get_file_name('validation', type))

    print('\nMaking Predictions on Test Input Images...')
    test_labels_pred = clf.predict(test_img)

    print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
    acc = accuracy_score(test_labels, test_labels_pred)

    print('\nCreating Confusion Matrix for Test Data...')
    conf_mat_test = confusion_matrix(test_labels, test_labels_pred)

    print('\nPredicted Labels for Test Images: ', test_labels_pred)
    print('\nAccuracy of Classifier on Test Images: ', acc)
    print('\nConfusion Matrix for Test Data:', conf_mat_test)

    plt.matshow(conf_mat_test)
    plt.title('Confusion Matrix for Test Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(get_file_name('test', type))

    a = np.random.randint(1, 50, 20)
    for idx, i in enumerate(a):
        two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
        plt.title('Original Label: {0}  Predicted Label: {1}'.format(
            test_labels[i], test_labels_pred[i]))
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        filename = image_file_name(idx, test_labels[i])
        plt.savefig(filename)
        plt.clf()

    print('Done')

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
