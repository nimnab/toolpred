import logging

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from HierarchicalClassifier import HierarchicalClassifier

logging.basicConfig(level=logging.DEBUG)

path = '/home/nnabizad/code/toolpred/ipythons/'

classifiers = [
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(kernel="rbf", gamma="auto", C=100, probability=True, class_weight='balanced'),
    GaussianNB(),
    ComplementNB(),
    tree.DecisionTreeClassifier(class_weight='balanced'),
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
    preds = model.predict(xtest=X_test, lens=lens)
    per, rec, f1 = per_recal(y_test,preds)
    print(per, rec, f1)



if __name__ == '__main__':
    layer = ['_gru_1', '_time_distributed_1', '_dense_2'][1]
    activations = ['relu', 'logistic', 'tanh']
    alpha = [0.001, 0.00001, 0.0001]
    hiddens = [256, 128, 64]
    class_hierarchy = np.load(path + 'svmdata/mactool_hi.pkl')
    X_train = np.load(path + 'svmdata/mactools_xtrain{}.npy'.format(layer))
    y_train = np.load(path + 'svmdata/mactools_ytrain{}.pkl'.format(layer))
    X_test = np.load(path + 'svmdata/mactools_xtest{}.npy'.format(layer))
    y_test = np.load(path + 'svmdata/mactools_ytest{}.pkl'.format(layer))
    index = 4
    myclassifier(index=index)
