import sys

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
import logging
logging.basicConfig(level=logging.DEBUG)
from utils.util import output

path = '/home/nnabizad/code/toolpred/ipythons/'

classifiers = [
    GaussianNB(),
    ComplementNB(),
    tree.DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced'),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1, max_iter=1000)
]

names = ["GaussianNB", "ComplementNB", "DecisionTreeClassifier", "RandomForestClassifier", "MLPClassifier"]


def per_recal(target, pred):
    tp = fp = fn = 0
    for i in range(len(target)):
        if (target[i] and pred[i]): tp += 1
        if (not target[i] and pred[i]): fp += 1
        if (target[i] and not pred[i]): fn += 1
    if tp == 0:
        if fn == 0 and fp == 0:
            return 1, 1, 1
        else:
            return 0, 0, 0
    else:
        per = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * (per * rec) / (per + rec)
        return per, rec, f1


def multilabel(strategy, hidden, alpha, act):
    mlb = MultiLabelBinarizer()

    _ = mlb.fit_transform(y_train + y_test)
    y_trainb = mlb.transform(y_train)
    y_testb = mlb.transform(y_test)
    # bclf = OneVsRestClassifier(classifiers[2])
    bclf = OneVsRestClassifier(MLPClassifier(solver='lbfgs',  activation=act, hidden_layer_sizes=(hidden, 2), random_state=1, max_iter=1000))
    clf = HierarchicalClassifier(
        base_estimator=bclf,
        class_hierarchy=class_hierarchy,
        algorithm="lcn", training_strategy=strategy,
        # feature_extraction="raw",
        mlb=mlb,
        use_decision_function=False
    )
    clf.fit(X_train, y_trainb[:, :])
    predictions = clf.predict_proba(X_test)
    ranks = []
    corrects = 0
    for y in range(len(y_testb)):
        rank = np.where(np.argsort(-predictions[y]) == np.argmax(y_testb[y]))
        if np.argmax(predictions[y]) == np.argmax(y_testb[y]): corrects += 1
        ranks.append(rank)
    thresholds = [1e-5, 13 - 4, 1e-3, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # output("actication:{}, hidden{}, alpha{}\n".format(act, hidden, alpha),
    #        filename='1search_result.csv', func='write')
    # for val in thresholds:
    #     precision = []
    #     recall = []
    #     f1 = []
    #     pred = predictions.copy()
    #
    #     pred[pred >= val] = 1
    #     pred[pred < val] = 0
    #
    #     for man in range(len(pred)):
    #         per, rec, f_1 = per_recal(y_testb[man], pred[man])
    #         precision.append(per)
    #         recall.append(rec)
    #         f1.append(f_1)
    #     output(
    #         "actication:{}, hidden{}, alpha{}, Val: {}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}, RANK:{}, accu:{}\n".format(
    #             act, hidden, alpha, val, np.mean(precision), np.mean(recall), np.mean(f1), np.mean(ranks), corrects/len(y_testb)),
    #         filename='0search_result.csv')
    print(strategy,corrects/len(y_testb))
    print()


if __name__ == '__main__':
    layer = ['_gru_1', '_time_distributed_1', '_dense_2'][2]
    activations = ['relu', 'logistic', 'tanh']
    alpha = [0.001, 0.00001, 0.0001]
    hiddens = [256, 128, 64]
    class_hierarchy = np.load(path + 'svmdata/mactool_hi.pkl')
    X_train = np.load(path + 'svmdata/mactools_xtrain{}.npy'.format(layer))
    y_train = np.load(path + 'svmdata/mactools_ytrain{}.pkl'.format(layer))
    X_test = np.load(path + 'svmdata/mactools_xtest{}.npy'.format(layer))
    y_test = np.load(path + 'svmdata/mactools_ytest{}.pkl'.format(layer))
    stras = ["exclusive", "inclusive", "less_inclusive", "siblings", "exclusive_siblings"]
    ac = int(sys.argv[1])

    h = int(sys.argv[2])
    for st in stras:
        multilabel(st,hiddens[h], 0.01, activations[ac])
