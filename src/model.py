import sys
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense

from .shared.mnist_loader import MNIST
from .shared.utils import (
    save,
    create_pickle,
)

from .log import Log

class Model(Log):
    def __init__(self, strategy):
        super().__init__()
        style.use("ggplot")
        self.strategy = strategy
        self.classifier = self.select_classifier_from_strategy(strategy)
        self.strategies = ['knn', 'svm', 'rfc', 'mlp', 'rnn', 'cnn']

    def _1_load_data(self):
        self.log("Loading Training Data...")
        data = MNIST()
        img_train, labels_train = data.load_training()
        self.train_img = np.array(img_train)
        self.train_labels = np.array(labels_train)

        self.log("Loading Testing Data...")
        img_test, labels_test = data.load_testing()
        self.test_img = np.array(img_test)
        self.test_labels = np.array(labels_test)

    def _2_split_into_train_and_validation_sets(self):
        self.log("Splitting data into train & validation sets...")
        return model_selection.train_test_split(
            self.train_img, self.train_labels, test_size=0.1
        )

    def _3_fit_and_measure_validation(self, x_train, x_validation, y_train, y_validation):
        if self.strategy == "rnn":
            x_train = x_train.reshape((-1, 28, 28))
            x_validation = x_validation.reshape((-1, 28, 28))

        self.classifier.fit(x_train, y_train)
        self.classifier = create_pickle(self.classifier, self.strategy)

        self.log("Calculating Accuracy of trained Classifier...")
        y_pred, confidence = None, None

        if self.strategy == "rnn":
            _, self.accuracy = self.classifier.evaluate(x_validation, y_validation)
            y_validation_pred_probs = self.classifier.predict(x_validation)
            y_validation_pred_classes = np.argmax(y_validation_pred_probs, axis=1)
            self.conf_mat = confusion_matrix(y_validation, y_validation_pred_classes)
            self.test_img = self.test_img.reshape((-1, 28, 28))
            test_labels_pred_probs = self.classifier.predict(self.test_img)
            self.test_labels_pred = np.argmax(test_labels_pred_probs, axis=1)
        else:
            confidence = self.classifier.score(x_validation, y_validation)
            y_pred = self.classifier.predict(x_validation)
            self.accuracy = accuracy_score(y_validation, y_pred)
            self.conf_mat = confusion_matrix(y_validation, y_pred)
            self.test_labels_pred = self.classifier.predict(self.test_img)

        self.log(f"\n\nTraining Confidence: \n{confidence:.2f}\nPredicted Values: {y_pred}\nAccuracy of Classifier on Validation Image Data: {self.accuracy}")

        self.generate_confusion_matrix(self.conf_mat, 'validation')

    def _4_evaluate_on_test_set(self):
        test_confusion_matrix = confusion_matrix(self.test_labels, self.test_labels_pred)

        self.generate_confusion_matrix(test_confusion_matrix, 'test')
        self.generate_images_with_predictions_for_review()

    def train(self):
        self.log("Loading MNIST Data...")
        self._1_load_data()
        x_train, x_validation, y_train, y_validation = self._2_split_into_train_and_validation_sets()
        self._3_fit_and_measure_validation(x_train, x_validation, y_train, y_validation)
        self._4_evaluate_on_test_set()
        self.log("Done")
        
    def generate_images_with_predictions_for_review(self):
        a = np.random.randint(1, 50, 20)
        for idx, i in enumerate(a):
            two_d = (np.reshape(self.test_img[i], (28, 28)) * 255).astype(np.uint8)
            plt.title(
                f"Original Label: {self.test_labels[i]}  Predicted Label: {self.test_labels_pred[i]}"
            )
            plt.imshow(two_d, interpolation="nearest", cmap="gray")
            filename = f'tmp/output/{self.strategy}-{idx}-labeled-{self.test_labels[i]}-predict-{self.test_labels_pred[i]}'
            plt.savefig(filename)
            plt.clf()

    def generate_confusion_matrix(self, matrix, dataset):
        self.log(f'\n\nGenerating Confusion Matrix: {dataset}\n{matrix}')
        plt.matshow(matrix)
        plt.title(f"Confusion Matrix for {dataset} Data")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        for (i, j), value in np.ndenumerate(matrix):
            plt.text(j, i, f'{value}', ha='center', va='center', color='red')
        filename = f'matrices/{self.timestamp}__{self.strategy}__{dataset}__confusion_matrix'
        plt.savefig(save(filename))
        plt.clf()


    def select_classifier_from_strategy(self, type):
        self.log(f'select_classifier_from_strategy: {type}')
        if type == "knn":
            self.log("KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
        elif type == "svm":
            self.log("SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        elif type == "rfc":
            self.log("RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif type == "mlp":
            self.log(
                "MLPClassifier with hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.001"
            )
            return MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=300,
                alpha=1e-4,
                solver="sgd",
                verbose=10,
                random_state=1,
                learning_rate_init=0.001,
            )
        elif type == "rnn":
            self.log(
                "RecurrentNeuralNetwork with 128, input_shape=(28, 28), activation='relu'), Dense(10, activation='softmax'"
            )
            clf = Sequential(
                [
                    SimpleRNN(128, input_shape=(28, 28), activation="relu"),
                    Dense(10, activation="softmax"),
                ]
            )
            clf.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return clf
        elif type == "cnn":
            self.log(
                "ConvolutionalNeuralNetwork with n_neighbors=5, algorithm='auto', n_jobs=10"
            )
            return
