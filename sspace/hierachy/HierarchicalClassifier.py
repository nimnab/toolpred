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

    def inverse_hierachy(self, node, level):
        # collect samples for non leafs
        nonleafs = {tool for tool in self.hierarchy[node] if not self.isleaf(tool)}
        revrese_hirachy = dict()
        for tool in self.hierarchy['<ROOT>']: revrese_hirachy[tool] = tool
        for parent in nonleafs:
            for child in self.hierarchy[parent]:
                if self.isleaf(child):
                    revrese_hirachy[child] = parent
                else:
                    revrese_hirachy[child] = parent
                    for grandchild in self.hierarchy[child]:
                        if self.isleaf(grandchild):
                            if level ==1:
                                revrese_hirachy[grandchild] = parent
                            elif level == 2:
                                revrese_hirachy[grandchild] = child
                        else:
                            print(grandchild)
        return revrese_hirachy

    def predict(self, xtest, lens, level = 3):
        ys = []
        for i,x in enumerate(xtest):
            step = []
            y = self.clfs['<ROOT>'].predict_proba(x.reshape(1, -1))[0]
            args = np.argsort(-y)
            c = 0
            while (len(step) < lens[i]):
                label = self.clfs['<ROOT>'].classes_[args[c]]
                c+=1
                if self.isleaf(label) or level<2:
                    # print(label)
                    step.append(label)
                    continue
                elif label in self.clfs:
                    _y = self.clfs[label].predict(x.reshape(1, -1))[0]
                    if self.isleaf(_y) or level <3:
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

    def f1_score(self, targets, preds, level = 3):
        targets = targets.copy()
        if level < 3:
            rootreverse = self.inverse_hierachy('<ROOT>', level)
            for s in range(len(targets)):
                targets[s] = [rootreverse[i] for i in targets[s] if i in rootreverse]
        tp = fp = fn = 0
        for i, step in enumerate(targets):
            for j in step:
                if j in preds[i]:
                    tp += 1
                else:
                    fn += 1
            for j in preds[i]:
                if j not in step:
                    fp += 1
        if tp == 0:
            if fn == 0 and fp == 0:
                return 1
            else:
                return 0
        else:
            per = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * (per * rec) / (per + rec)
            return f1
