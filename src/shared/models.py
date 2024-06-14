from sklearn import model_selection, svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

from sklearn.metrics import accuracy_score, confusion_matrix

def get_model_type(type):
    if type == 'KNN':
        print('\nKNearestNeighbors with n_neighbors = 5, algorithm = auto, n_jobs = 10')
        return KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)
    elif type == 'SVM':
        print("\nSupportVectorMachines with gamma=0.1, kernel='poly'")
        return svm.SVC(gamma=0.1, kernel='poly')
    elif type == 'RFC':
        print("\nRandomForestClassifier with n_estimators=100, random_state=42")
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif type == 'MLP':
        print("\nMLPClassifier with hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=.1")
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                            solver='sgd', verbose=10, random_state=1,
                            learning_rate_init=.1)
    elif type == 'CNN':
        print("\nConvolutionalNeuralNetwork with n_neighbors=5, algorithm='auto', n_jobs=10")
        return
    elif type == 'RNN':
        print("\nRecurrentNeuralNetwork with 128, input_shape=(28, 28), activation='relu'), Dense(10, activation='softmax'")
        clf = Sequential([
            SimpleRNN(128, input_shape=(28, 28), activation='relu'),  # timesteps=28, input_dim=28
            Dense(10, activation='softmax')
        ])
        clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return clf
