import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import model_selection, svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
        self.log("Loading Data...")
        data = MNIST()
        self.log("Loading Training Data...")
        imgs_train, labels_train = data.load_training()
        self.train_imgs = np.array(imgs_train)
        self.train_labels = np.array(labels_train)

        self.log("Loading Testing Data...")
        imgs_test, labels_test = data.load_testing()
        self.test_imgs = np.array(imgs_test)
        self.test_labels = np.array(labels_test)

    def _2_split_into_train_and_validation_sets(self):
        self.log("Splitting Data into train & validation sets...")
        return model_selection.train_test_split(
            self.train_imgs, self.train_labels, test_size=0.1
        )

    def _3_fit_and_measure_validation(self, x_train, x_validation, y_train, y_validation):
        if self.strategy == "rnn":
            x_train = x_train.reshape((-1, 28, 28))
            x_validation = x_validation.reshape((-1, 28, 28))
            self.test_imgs = self.test_imgs.reshape((-1, 28, 28))

        self.classifier.fit(x_train, y_train)
        self.classifier = create_pickle(self.classifier, self.strategy)

        self.log("Calculating Accuracy of trained Classifier...")
        y_pred, confidence = None, None

        if self.strategy == "rnn":
            _, accuracy = self.classifier.evaluate(
                x_validation, y_validation)
            y_pred = self.classifier.predict(x_validation)
            y_pred = np.argmax(y_pred, axis=1)
            conf_matrix = confusion_matrix(y_validation, y_pred)
            test_labels_pred_probs = self.classifier.predict(self.test_imgs)
            self.test_labels_pred = np.argmax(test_labels_pred_probs, axis=1)
        else:
            confidence = self.classifier.score(x_validation, y_validation)
            y_pred = self.classifier.predict(x_validation)
            accuracy = accuracy_score(y_validation, y_pred)
            conf_matrix = confusion_matrix(y_validation, y_pred)
            self.test_labels_pred = self.classifier.predict(self.test_imgs)

        report_str = classification_report(y_validation, y_pred)
        self.log(f"\nClassification Report: \n{report_str}")
        self.log(f"\n\nTraining Confidence: \n{confidence:.2f}\nAccuracy: {accuracy:.2f}\nPredicted Values: {y_pred}\n")
        self.generate_confusion_matrix(conf_matrix, 'validation')

    def _4_evaluate_on_test_set(self):
        test_confusion_matrix = confusion_matrix(
            self.test_labels, self.test_labels_pred)

        self.generate_confusion_matrix(test_confusion_matrix, 'test')
        self.generate_images_with_predictions_for_review()

    def train(self):
        self.log("Training Starting...")
        self._1_load_data()
        x_train, x_validation, y_train, y_validation = self._2_split_into_train_and_validation_sets()
        self._3_fit_and_measure_validation(
            x_train, x_validation, y_train, y_validation)
        self._4_evaluate_on_test_set()
        self.render_scatter_plot_with_decision_boundaries()
        self.log("Done")

    def generate_images_with_predictions_for_review(self):
        a = np.random.randint(1, 50, 20)
        for idx, i in enumerate(a):
            two_d = (np.reshape(
                self.test_imgs[i], (28, 28)) * 255).astype(np.uint8)
            plt.title(f"Original Label: {self.test_labels[i]} Predicted Label: {self.test_labels_pred[i]}")
            plt.imshow(two_d, interpolation="nearest", cmap="gray")
            filename = f'tmp/output/{self.strategy}-{idx}-labeled-{self.test_labels[i]}-predict-{self.test_labels_pred[i]}'
            plt.savefig(filename)
            plt.clf()

    def generate_confusion_matrix(self, matrix, dataset):
        self.log(f'\nGenerating Confusion Matrix: {dataset}\n{matrix}\n')
        plt.matshow(matrix)
        plt.suptitle(f"Confusion Matrix", fontsize=16)
        plt.title(f'Data: {dataset.capitalize()}\nStrategy: {self.strategy.upper()}', fontsize=8)

        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        for (i, j), value in np.ndenumerate(matrix):
            plt.text(j, i, f'{value}', ha='center', va='center', color='white')
        filename = f'matrices/{self.timestamp}__{self.strategy}__{dataset}__confusion_matrix'
        plt.savefig(save(filename))
        plt.clf()

    def render_scatter_plot_with_decision_boundaries(self):
        print("Shape of self.train_imgs:", self.train_imgs.shape)
        print("Shape of self.test_imgs:", self.test_imgs.shape)
        
        train_img_flat = self.train_imgs.reshape(len(self.train_imgs), -1)
        test_imgs_flat = self.test_imgs.reshape(len(self.test_imgs), -1)
        
        scaler = StandardScaler()
        train_img_flat = scaler.fit_transform(train_img_flat)
        test_imgs_flat = scaler.transform(test_imgs_flat)
        
        pca = PCA(n_components=2)
        x_train_pca = pca.fit_transform(train_img_flat)
        x_test_pca = pca.transform(test_imgs_flat)

        # Determine dynamic limits for x and y axes
        x_min = min(np.min(x_train_pca[:, 0]), np.min(x_test_pca[:, 0]))
        x_max = max(np.max(x_train_pca[:, 0]), np.max(x_test_pca[:, 0]))
        y_min = min(np.min(x_train_pca[:, 1]), np.min(x_test_pca[:, 1]))
        y_max = max(np.max(x_train_pca[:, 1]), np.max(x_test_pca[:, 1]))

        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

        for ax, weights in zip(axs, ("uniform", "distance")):
            self.classifier.set_params(weights=weights)
            self.classifier.fit(x_train_pca, self.train_labels)
            
            disp = DecisionBoundaryDisplay.from_estimator(
                self.classifier,
                x_train_pca,
                response_method="predict",
                plot_method="pcolormesh",
                shading="auto",
                ax=ax,
            )

            scatter = ax.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=self.test_labels, edgecolors="k")
            legend = ax.legend(
                scatter.legend_elements()[0],
                np.unique(self.test_labels),
                loc="lower left",
                title="Classes",
            )

            # Set dynamic limits for x and y axes
            ax.set_xlim(x_min - 0.1 * abs(x_min), x_max + 0.1 * abs(x_max))
            ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
            
            ax.set_title(f"KNN decision boundaries\n(weights={weights!r})")

        plt.tight_layout()
        plt.savefig(save(f'{self.strategy}-xlim-ylim-graph'))

    def select_classifier_from_strategy(self, strategy):
        self.log(f'select_classifier_from_strategy: {strategy}')
        if strategy == "knn":
            self.log(
                "KNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10")
            return KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=10)
        elif strategy == "svm":
            self.log("SupportVectorMachines with gamma=0.1, kernel='poly'")
            return svm.SVC(gamma=0.1, kernel="poly")
        elif strategy == "rfc":
            self.log(
                "RandomForestClassifier with n_estimators=100, random_state=42")
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif strategy == "mlp":
            self.log(
                "MLPClassifier with hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.001"
            )
            return MLPClassifier(
                hidden_layer_sizes=(100),
                max_iter=300,
                alpha=1e-4,
                solver="sgd",
                verbose=10,
                random_state=1,
                learning_rate_init=0.001,
            )
        elif strategy == "rnn":
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
        elif strategy == "cnn":
            self.log(
                "ConvolutionalNeuralNetwork with n_neighbors=5, algorithm='auto', n_jobs=10"
            )
            return
