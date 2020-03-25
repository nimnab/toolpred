import logging
import sys

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from HierarchicalClassifier import HierarchicalClassifier

logging.basicConfig(level=logging.DEBUG)

path = '/home/nnabizad/code/toolpred/ipythons/'

classifiers = [
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(kernel="rbf", gamma="auto", C=100, probability=True, class_weight='balanced'),
    GaussianNB(),
    ComplementNB(),
    tree.DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced'),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=1000)
]

names = ["linear", "rbf", "GaussianNB", "ComplementNB", "DecisionTreeClassifier", "RandomForestClassifier",
         "MLPClassifier"]


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


def myclassifier(index):
    # bclf = OneVsRestClassifier(classifiers[2])
    sclf = classifiers[index]
    model = HierarchicalClassifier(hierarchy=class_hierarchy)
    rootleafs, rootnonleafs, rootreverse = model.node_data('<ROOT>')
    xtrain, ytrain = model.rolled_data(rootleafs, rootreverse, X_train, y_train)
    xtest, ytest = model.rolled_data(rootleafs, rootreverse, X_test, y_test)
    # mlp = MLPClassifier(max_iter=1000)
    # svm = SVC(kernel="rbf", probability=True, class_weight='balanced')
    # mlpparameter_space = {
    #     'hidden_layer_sizes': [(200, 100, 50,), (200, 50), (100, 50,), (50,), (100,), (200,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam', 'lbfgs'],
    #     'alpha': [1e-5, 0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # svmparameters = {'gamma': ["auto", 1e-20, 1e-10, 1e-5, 0.001, 0.01, 1], "C": [100, 1000, 0.1, 0.01]}
    treepars = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'class_weight': ['balanced', None],
                'min_samples_split': [2, 3, 4, 5, 10], 'min_samples_leaf': [1, 2, 3, 4, 5, 10],
                'min_weight_fraction_leaf': [0, 0, 1e-5, 0.001, 0.01], 'max_features': ['auto', 'sqrt', 'log2', None],
                'max_leaf_nodes': [None, 1, 2, 3, 4, 5], 'min_impurity_decrease': [0, 0, 1e-5, 0.001, 0.01],
                'min_impurity_split': [0, 0, 1e-10, 1e-5, 0.001, 0.01]}
    clf = GridSearchCV(sclf, treepars, n_jobs=-1, cv=3)
    clf.fit(xtrain+xtest, ytrain+ytest)
    # preds = clf.predict(xtest)
    # accuracy = sum(1 for x, y in zip(preds, ytest) if x == y) / len(ytest)
    # print(layer, names[index], accuracy)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("parts%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


if __name__ == '__main__':
    data = 'macparts'
    layer = ['_gru_1', '_time_distributed_1', '_dense_2'][2]
    activations = ['relu', 'logistic', 'tanh']
    alpha = [0.001, 0.00001, 0.0001]
    hiddens = [256, 128, 64]
    class_hierarchy = np.load(path + 'svmdata/{}_hi.pkl'.format(data))
    X_train = np.load(path + 'svmdata/{}_xtrain{}.npy'.format(data,layer))
    y_train = np.load(path + 'svmdata/{}_ytrain{}.pkl'.format(data,layer))
    X_test = np.load(path + 'svmdata/{}_xtest{}.npy'.format(data,layer))
    y_test = np.load(path + 'svmdata/{}_ytest{}.pkl'.format(data,layer))
    index = 4
    myclassifier(index=index)
