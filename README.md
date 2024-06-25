# 12 Machine Learning Models for Image Recognition

- 12 image recognition using 6 different algorithms
- 6 different algorithms
- 6 package models
- 6 custom models

## Getting Started

- Install dependencies

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run model, appending time logs for comparison

```sh
(time python src/main.py KNN) >> tmp/logs/KNN-summary.log 2>&1
(time python src/main.py SVM) >> tmp/logs/SVM-summary.log 2>&1
(time python src/main.py RFC) >> tmp/logs/RFC-summary.log 2>&1
(time python src/main.py MLP) >> tmp/logs/MLP-summary.log 2>&1
(time python src/main.py RNN) >> tmp/logs/RNN-summary.log 2>&1
(time python src/main.py CNN) >> tmp/logs/CNN-summary.log 2>&1

(time python src/main.py KNN) >> tmp/logs/KNN-summary.log 2>&1 &&
(time python src/main.py SVM) >> tmp/logs/SVM-summary.log 2>&1 &&
(time python src/main.py RFC) >> tmp/logs/RFC-summary.log 2>&1 &&
(time python src/main.py MLP) >> tmp/logs/MLP-summary.log 2>&1 &&
(time python src/main.py RNN) >> tmp/logs/RNN-summary.log 2>&1 &&
(time python src/main.py CNN) >> tmp/logs/CNN-summary.log 2>&1

(time python src/main.py KNN) >> tmp/logs/KNN-summary.log 2>&1 && (time python src/main.py RNN) >> tmp/logs/RNN-summary.log 2>&1
```
