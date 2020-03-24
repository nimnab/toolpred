import sys

import numpy as np
from gensim import matutils

from utils.datas import Data
from utils.util import output
from .chain import Chain_withid

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
sentsembs = dict()

def cossim(vec1, vec2):
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    return sim



class Markov_mart():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        self.table = self.creat_table()
        self.write_result()

        
        
    def write_result(self):
        maxsts = [max([len(i) for i in self.mydata.train]),1,2,3,4,5]
        for maxst in maxsts:
            self.maxst = maxst
            self.models = [Chain_withid(self.mydata.train, i).model for i in range(0, self.maxst)]
            preds, acc = self.accu_all(self.mydata.test)
            print('{}, {}, {}'.format(seed, self.maxst, acc))
            output('{}, {}, {}'.format(seed, self.maxst, acc), filename=filename, func='write')


    # 
    def similarity_score(self,id, tool):
        if tool in self.table:
            prob = 1
            for w in self.mydata.titles_test[id]:
                prob *= self.table[tool][w]
            return prob
        else:
            return 0

    
    def predict(self, lis, id):
        history = tuple(lis)
        order = (self.maxst-1) if len(lis) > (self.maxst-1) else len(lis)
        if history in self.models[order].keys():
            normp = sum([self.models[order][history][k][0] for k in self.models[order][history].keys()])
            # norms = sum([similarity_score(id,self.models[order][history][k][1]) for k in self.models[order][history].keys()])
            self.prediction = max(self.models[order][history].keys(), key=(lambda k:  (self.models[order][history][k][0]/normp) * self.similarity_score(id, k)))
            return self.prediction
        elif len(lis) > 0:
            lis2 = lis[1:]
            self.predict(lis2, id)
        else:
            self.prediction = -1
            return None
    
    
    def accu_all(self,test):
        corr = total = 0
        preds = []
        for id, manual in enumerate(test):
            # print(id)
            tmppred = []
            oldtool = [1]
            for tool in manual[1:]:
                self.predict(oldtool, id)
                total += 1
                if self.prediction == tool:
                    corr +=1
                oldtool.append(tool)
                tmppred.append(self.prediction)
            preds.append(tmppred)
        return preds , (corr)/(total)


    def creat_table(self):
        table = dict()
        for man, title in zip(self.mydata.train, self.mydata.titles_train):
            for t in man:
                if t not in table:
                    table[t] = dict()
                for w in title:
                    if w in table[t] : table[t][w]+=1
                    else : table[t][w] = 1
        trainvocab = set([item for sublist in self.mydata.titles_train for item in sublist])
        testvocab = set([item for sublist in self.mydata.titles_test for item in sublist])
        vocab = trainvocab.union(testvocab)

        for tool in table:
            zeros = len(vocab)- len(table[tool].keys()&vocab)
            nonzeros = len(vocab) - zeros
            summ = sum(table[tool].values())
            for word in vocab:
                if word in table[tool]:
                    table[tool][word] =  (table[tool][word]-alpha)/summ
                else:
                    table[tool][word] = ((alpha * nonzeros) / zeros) / summ
        return table





if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/res/Emarkov_bow.csv'
    alpha = 1
    seed = int(sys.argv[1])
    # order = int(sys.argv[1])
    simdic = dict()
    print('Training with seed:{}'.format(seed), flush=True)
    Markov_mart(extracted=True)
    sys.exit()

