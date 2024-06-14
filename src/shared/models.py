from sklearn import model_selection, svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def get_model_type(type):
    if type == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)
    elif type == 'SVM':
        return svm.SVC(gamma=0.1, kernel='poly')
    elif type == 'RFC':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif type == '':
        return
