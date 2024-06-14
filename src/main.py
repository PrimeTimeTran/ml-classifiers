import sys

from _1_knn.k_nearest_neighbors import model as knn_model
from _2_svm.support_vector_machines import model as svm_model
from _3_rfc.random_forest_classifier import model as rfc_model
from _4_mlp.multilayer_perceptron import model as mlp_model
from _5_rnn.recurrent_neural_networks import model as rnn_model
from _6_cnn.convolution_neural_networks import model as cnn_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    model_name = sys.argv[1].upper()

    if model_name == 'KNN':
        knn_model('KNN')
    elif model_name == 'SVM':
        svm_model('SVM')
    elif model_name == 'RFC':
        rfc_model('RFC')
    elif model_name == 'MLP':
        mlp_model('MLP')
    elif model_name == 'RNN':
        rnn_model('RNN')
    elif model_name == 'CNN':
        cnn_model('CNN')
    else:
        print(f"Unknown model '{model_name}'. Please choose one of: KNN, SVM, RFC, MLP, RNN, CNN")
        sys.exit(1)
