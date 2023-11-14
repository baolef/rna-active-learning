# Created by Baole Fang at 10/31/23
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import pickle
import xgboost
import pandas as pd
import matplotlib.pyplot as plt

from learning import active_learning


def load(path: str, percentage: int):
    data = np.load(path)
    X = data['X']
    Y = data['Y']
    percentile = np.percentile(Y, percentage)
    Y = Y > percentile
    return X, Y.astype(int)


def main(args):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    config = {'n_jobs': os.cpu_count(), 'use_label_encoder': False, 'eval_metric': 'logloss'}
    X, y = load(args.input, args.percentage)
    X_train, X_test, y_train, y_test = train_test_split(X[:100000], y[:100000], test_size=args.test,
                                                        random_state=args.seed)
    model = xgboost.XGBClassifier(**config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).astype(int)
    report = classification_report(y_test, y_pred)
    path = os.path.join(args.output, os.path.basename(args.input).split('.')[0], args.model, 'prediction')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write(report)
    df = pd.DataFrame(cm)
    df.to_csv(os.path.join(path, 'confusion.csv'), index=False, header=False)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(os.path.join(path, 'confusion.png'))

    active_learning([X_train, X_test, y_train, y_test], xgboost.XGBClassifier, config, path, 1000, 9000, 100, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the classification problem')
    parser.add_argument('-i', '--input', help='path of the data', type=str, required=True)
    parser.add_argument('-t', '--test', help='test set size', type=float, default=0.2)
    parser.add_argument('-o', '--output', help='path of output model', type=str, default='outputs')
    parser.add_argument('-m', '--model', help='model name', type=str, default='xgboost')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    parser.add_argument('-p', '--percentage', help='percentage of positive samples', type=int, default=50)

    args = parser.parse_args()
    main(args)
