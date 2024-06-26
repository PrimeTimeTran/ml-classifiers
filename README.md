# Handwritten Digit Recognition with Machine Learning

- 6 different algorithms
- KNN, SVM, RFC, MLP, RNN, CNN

![demo](./tmp/knn_scatter-plot.png)
![demo](./tmp/svm_scatter-with_boundaries.png)

## Getting Started

- Install dependencies

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Use `-s` to select a strategy for training. The available strategies are KNN, SVM, RFC, MLP, RNN, CNN

```sh
python main.py -s knn
```

Use `-m` to select a method to run. For example with `init_trained_model` you can pick up training from where you left off with a saved pickle inside of `tmp/models`

```sh
python main.py -m init_trained_model
```

## Renders

Confusion matrices of the results of training using KNN as well as summaries of other models & their strategies.

![demo](./tmp/knn_test_conf_matrix.png)
![demo](./tmp/knn_train_conf_matrix.png)
![demo](./tmp/mlp_neural-network.png)
![demo](./tmp/rfc_decision-tree.png)
![demo](./tmp/rnn_neural-network.png)

## References

- [Classification Strategies - Scikit-Learn](https://scikit-learn.org/stable/)
- [Charts - Matplotlib](https://python-charts.com/matplotlib/)
