import sys

from _1_knn.k_nearest_neighbors import model as knn_model
from _2_svm.support_vector_machines import model as svm_model
from _3_rfc.random_forest_classifier import model as rfc_model
from _4_mlp.multilayer_perceptron import model as mlp_model
from _5_rnn.recurrent_neural_networks import model as rnn_model
from _6_cnn.convolution_neural_networks import model as cnn_model

if __name__ == "__main__":
    knn_model('KNN')
    svm_model('SVM')
    rfc_model('RFC')
    mlp_model('MLP')
    rnn_model('RNN')
    # cnn_model('CNN')
