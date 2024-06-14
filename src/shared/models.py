from sklearn import model_selection, svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix


def get_model_type(type):
    if type == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)
    elif type == 'SVM':
        return svm.SVC(gamma=0.1, kernel='poly')
    elif type == 'RFC':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif type == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                            solver='sgd', verbose=10, random_state=1,
                            learning_rate_init=.1)
    elif type == 'CNN':
        return
    elif type == 'RNN':
        return
