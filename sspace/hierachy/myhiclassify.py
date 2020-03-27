import logging

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sys

# from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from HierarchicalClassifier import HierarchicalClassifier

# logging.basicConfig(level=logging.DEBUG)

path = '/home/nnabizad/code/toolpred/ipythons/'

classifiers = [
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(kernel="rbf", gamma="auto", C=100, probability=True, class_weight='balanced'),
    GaussianNB(),
    ComplementNB(),
    tree.DecisionTreeClassifier(min_samples_split=10),
    RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced'),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=1000)
]

names = ["linear", "rbf", "GaussianNB", "ComplementNB", "DecisionTreeClassifier", "RandomForestClassifier",
         "MLPClassifier"]


def per_recal(targets, preds):
    tp = fp = fn = 0
    for i,step in enumerate(targets):
        for j in step:
            if j in preds[i]:
                tp+=1
            else:
                fn+=1
        for j in preds[i]:
            if j not in step:
                fp+=1
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


def myclassifier(index):
    # bclf = OneVsRestClassifier(classifiers[2])
    clf = classifiers[index]
    model = HierarchicalClassifier(clf=clf, hierarchy=class_hierarchy)
    model.fit(xtrain=X_train, ytrain=y_train)
    lens = [len(a) for a in y_test]
    for level in (1,2,3):
        preds = model.predictml(xtest=X_test, lens=lens, level=level)
        f1 = model.f1_score(y_test,preds, level=level)
        print(layer, names[index], f1)



if __name__ == '__main__':
    # data = 'mactools'
    data = 'macparts'
    layer = ['_time_distributed_1', '_dense_2'][2]
    activations = ['relu', 'tanh', '']
    alpha = [0.001, 0.00001, 0.0001]
    hiddens = [256, 128, 64]
    class_hierarchy = np.load(path + 'svmdata/{}_hi.pkl'.format(data))
    X_train = np.load(path + 'svmdata/{}_xtrain{}.npy'.format(data,layer))
    y_train = np.load(path + 'svmdata/{}_ytrain{}.pkl'.format(data,layer))
    X_test = np.load(path + 'svmdata/{}_xtest{}.npy'.format(data,layer))
    y_test = np.load(path + 'svmdata/{}_ytest{}.pkl'.format(data,layer))
    index = int(sys.argv[1])
    myclassifier(index=index)
