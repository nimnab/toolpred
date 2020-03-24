import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier

from utils.util import output

path = '/ipythons/'

classifiers = [
    # LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',class_weight='balanced'),
    # svm.LinearSVC(class_weight='balanced'),
    SVC(kernel="rbf", gamma="auto", C=100, probability=True,class_weight='balanced'),
    RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0,class_weight='balanced'),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ]

names = ["SVM","Random Forest", "Neural Net"]



# y_strain = [y[0] for y in y_train]
# y_stest = [y[0] for y in y_test]

def classifysearch():
    for name, clf in zip(names, classifiers):
        hclf = HierarchicalClassifier(
            feature_extraction="raw",
            algorithm= "lcn",
            training_strategy= "less_inclusive",
            base_estimator=clf,
            class_hierarchy=class_hierarchy,
            # prediction_depth = "nmlnp"
            #         mlb=mlb,
            #         mlb_prediction_threshold = 0.2
        )
        hclf.fit(X_train, y_strain)
        preds = hclf.predict([x.reshape(1,-1) for x in X_test])
        accuracy = sum(1 for x, y in zip(preds, y_stest) if x == y) / len(y_stest)
        print(name, accuracy)


def svc_param_selection(s,C):
    hclf = HierarchicalClassifier(
        # use_decision_function=True,
        # root="<ROOT>",
        feature_extraction="raw",
        algorithm="lcn",
        training_strategy=s,
        base_estimator=SVC(kernel="rbf", gamma=C, C=100, probability=True),
        class_hierarchy=class_hierarchy
        # prediction_depth = "nmlnp"
        #         mlb=mlb,
        #         mlb_prediction_threshold = 0.2
    )
    hclf.fit(X_train, y_strain)
    preds = hclf.predict([x.reshape(1,-1) for x in X_test])
    accuracy = sum(1 for x, y in zip(preds, y_stest) if x == y) / len(y_stest)
    print(layer, C,s, accuracy)


def multilabel(strategy,gamma, coef):

    mlb = MultiLabelBinarizer()

    _ = mlb.fit_transform(y_train+y_test)
    y_trainb = mlb.transform(y_train)
    y_testb = mlb.transform(y_test)
    # bclf = OneVsRestClassifier(LinearSVC(class_weight='balanced'))
    bclf = OneVsRestClassifier(SVC(kernel="rbf", gamma=gamma, C=coef, probability=True,class_weight='balanced'))
    clf = HierarchicalClassifier(
        base_estimator=bclf,
        class_hierarchy=class_hierarchy,
        algorithm="lcn", training_strategy=strategy,
        feature_extraction="raw",
        mlb=mlb,
        use_decision_function=True
    )
    clf.fit(X_train, y_trainb[:, :])
    preds = clf.predict_proba(X_test)
    ranks = []
    for y in range(len(y_testb)):
        rank = np.where(np.argsort(-preds[y]) == np.argmax(y_testb[y]))
        ranks.append(rank)
    output(", {} , {} , {},  {}".format(strategy,gamma, coef,np.mean(ranks)), filename=layer+'_result.csv', func='write')


if __name__ == '__main__':
    layer = '_dense_2'
    class_hierarchy = np.load(path + 'svmdata/mactool_hi.pkl')
    X_train = np.load(path + 'svmdata/mactools_xtrain{}.npy'.format(layer))
    y_train = np.load(path + 'svmdata/mactools_ytrain{}.pkl'.format(layer))
    X_test = np.load(path + 'svmdata/mactools_xtest{}.npy'.format(layer))
    y_test = np.load(path + 'svmdata/mactools_ytest{}.pkl'.format(layer))
    stras = ["exclusive", "inclusive", "less_inclusive", "siblings", "exclusive_siblings"]
    gamma= float(sys.argv[1])
    coef= float(sys.argv[2])
    for s in stras:
        multilabel(s,gamma, coef)