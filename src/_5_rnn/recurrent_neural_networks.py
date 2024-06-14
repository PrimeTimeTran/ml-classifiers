import os
import sys
import pickle
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from shared.mnist_loader import MNIST
from shared.models import get_model_type
from shared.utils import setup_save_directory, create_log_file, image_file_name, get_file_name, create_pickle, load_mnist_images, load_mnist_labels

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
    train_img = train_img.reshape((-1, 28, 28))
    train_labels = np.array(labels_train)

    print('\nLoading Testing Data...')
    img_test, labels_test = data.load_testing()
    test_img = np.array(img_test)
    test_img = test_img.reshape((-1, 28, 28))
    test_labels = np.array(labels_test)

    x = train_img
    y = train_labels

    print('\nPreparing Classifier Training and Validation Data...')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    x_train = x_train.reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28, 28))


    print(f'Shape of X_train: {x_train.shape}')
    print(f'Shape of X_val: {x_test.shape}')

    # Get model (assuming get_model_type returns a Keras model)
    clf = get_model_type('RNN')

    # Train model
    clf.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Save model
    clf = create_pickle(clf, type)

    # Evaluate on validation data
    loss, accuracy = clf.evaluate(x_test, y_test)
    print(f'Validation accuracy: {accuracy}')

    # Predict on validation data
    y_test_pred_probs = clf.predict(x_test)
    y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)

    # Compute accuracy and confusion matrix for validation data
    accuracy_test = accuracy_score(y_test, y_test_pred_classes)
    conf_mat_val = confusion_matrix(y_test, y_test_pred_classes)
    print(f'Validation Accuracy: {accuracy_test}')
    print(f'Validation Confusion Matrix:\n{conf_mat_val}')

    # Plot confusion matrix for validation data
    plt.matshow(conf_mat_val)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(get_file_name('validation_confusion_matrix', type))
    plt.clf()

    # Load testing data
    img_test, labels_test = data.load_testing()
    test_img = np.array(img_test)
    test_labels = np.array(labels_test)
    test_img = test_img.reshape((-1, 28, 28))

    # Predict on testing data
    test_labels_pred_probs = clf.predict(test_img)
    test_labels_pred_classes = np.argmax(test_labels_pred_probs, axis=1)

    # Compute accuracy and confusion matrix for testing data
    acc = accuracy_score(test_labels, test_labels_pred_classes)
    conf_mat_test = confusion_matrix(test_labels, test_labels_pred_classes)

    # Print and save results
    print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
    print(f'Accuracy of Classifier on Test Images: {acc}')
    print(f'Confusion Matrix for Test Data:\n{conf_mat_test}')

    plt.matshow(conf_mat_test)
    plt.title('Confusion Matrix for Test Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(get_file_name('test_confusion_matrix', type))
    plt.clf()

    # Plot some sample images with original and predicted labels
    a = np.random.randint(1, 50, 20)
    for idx, i in enumerate(a):
        two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
        plt.title(f'Original Label: {test_labels[i]}  Predicted Label: {test_labels_pred_classes[i]}')
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        filename = image_file_name(idx, test_labels[i])
        plt.savefig(filename)
        plt.clf()

    print('Done')

if __name__ == '__main__':
    model('RNN')
