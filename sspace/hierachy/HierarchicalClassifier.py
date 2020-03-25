from sklearn.base import clone
import numpy as np


class HierarchicalClassifier():
    def __init__(self,
                 hierarchy,
                 clf= None):
        self.clf = clf
        self.clfs = dict()
        self.hierarchy = hierarchy

    def fit(self, xtrain, ytrain):
        rootleafs, rootnonleafs, rootreverse = self.node_data('<ROOT>')
        rootx, rooty = self.rolled_data(rootleafs, rootreverse, xtrain, ytrain)
        rclf = clone(self.clf)
        rclf.fit(rootx, rooty)
        self.clfs['<ROOT>'] = rclf
        for nleaf in rootnonleafs:
            leafs, nonleafs, revh = self.node_data(nleaf)
            _x, _y = self.rolled_data(leafs, revh, xtrain, ytrain)
            if _x:
                _clf = clone(self.clf)
                _clf.fit(_x, _y)
                self.clfs[nleaf] = _clf
            for _nleaf in nonleafs:
                _leafs, _nonleafs, _revh = self.node_data(_nleaf)
                _xt, _yt = self.rolled_data(_leafs, _revh, xtrain, ytrain)
                if _xt:
                    _clft = clone(self.clf)
                    _clft.fit(_xt, _yt)
                    self.clfs[_nleaf] = _clft

    def rolled_data(self, leafs, revrese_hirachy, xtrain, ytrain):
        x = []
        y = []
        for i in range(len(ytrain)):
            for j in ytrain[i]:
                if j in leafs:
                    y.append(j)
                    x.append(xtrain[i])
                elif j in revrese_hirachy:
                    y.append(revrese_hirachy[j])
                    x.append(xtrain[i])
        return x, y

    def node_data(self, node):
        # collect samples for non leafs
        leafs = {tool for tool in self.hierarchy[node] if self.isleaf(tool)}
        nonleafs = {tool for tool in self.hierarchy[node] if not self.isleaf(tool)}
        revrese_hirachy = dict()
        for parent in nonleafs:
            for child in self.hierarchy[parent]:
                if self.isleaf(child):
                    revrese_hirachy[child] = parent
                else:
                    for grandchild in self.hierarchy[child]:
                        if self.isleaf(grandchild):
                            revrese_hirachy[grandchild] = parent
                        else:
                            print(grandchild)
        return leafs, nonleafs, revrese_hirachy

    def predict(self, xtest, lens):
        ys = []
        for i,x in enumerate(xtest):
            step = []
            y = self.clfs['<ROOT>'].predict_proba(x.reshape(1, -1))[0]
            args = np.argsort(-y)
            c = 0
            while (len(step) < lens[i]):
                label = self.clfs['<ROOT>'].classes_[args[c]]
                c+=1
                if self.isleaf(label):
                    # print(label)
                    step.append(label)
                    continue
                elif label in self.clfs:
                    _y = self.clfs[label].predict(x.reshape(1, -1))[0]
                    if self.isleaf(_y):
                        step.append(_y)
                        continue
                    elif _y in self.clfs:
                        _yt = self.clfs[_y].predict(x.reshape(1, -1))[0]
                        if self.isleaf(_yt):
                            step.append(_yt)
            ys.append(step)
        return ys


    def isleaf(self, node):
        if node in self.hierarchy.keys():
            return False
        else:
            return True
