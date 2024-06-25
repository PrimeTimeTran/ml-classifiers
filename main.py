import argparse

from src.model import Model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--strategy", type=str, default='knn')
    ap.add_argument("-mi", "--method", type=str, default='train')
    args = vars(ap.parse_args())
    strategy = args['strategy']
    method = args['method']

    model = Model(strategy)

    if method == 'train':
      model.train()
