from shared.mnist_loader import MNIST
from shared.models import get_model_type,base_model
from shared.utils import setup_save_directory, create_log_file, image_file_name, get_file_name, create_pickle, load_mnist_images, load_mnist_labels

def model(type):
    clf = get_model_type(type)
    base_model(type, clf)
