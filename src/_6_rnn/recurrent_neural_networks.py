import os
import sys
import pickle
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

from sklearn.metrics import accuracy_score, confusion_matrix

from shared.mnist_loader import MNIST
from shared.utils import setup_save_directory, create_log_file, image_file_name, get_file_name, create_pickle

def model(type):
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

    print("\nRecurrentNeuralNetwork with 128, input_shape=(28, 28), activation='relu'), Dense(10, activation='softmax'")
    clf = Sequential([
        SimpleRNN(128, input_shape=(28, 28), activation='relu'),  # timesteps=28, input_dim=28
        Dense(10, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(x_train, y_train, epochs=10)

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
