import os
import shutil

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, '../../tmp/output')
logs_dir = os.path.join(base_dir, '../../tmp/logs')
data_dir = os.path.join(base_dir, '../../tmp/dataset')

conf_matrix_test_filename = os.path.join(
    save_dir, 'confusion-matrix-for-test-data.png')
conf_matrix_validation_filename = os.path.join(
    save_dir, 'confusion-matrix-for-validation-data.png')

def setup_save_directory():
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

def image_file_name(idx, value):
    return os.path.join(
        save_dir, f'{idx}-original-{value}-predict-{value}.png')

def create_log_file(name):
    return open(f'{logs_dir}/{name}', "w")
