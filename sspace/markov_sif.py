import sys

import numpy as np
from fse import IndexedList
from fse.models import SIF
from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors

# from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.util import output
from .chain import Chain_withid

# from fse import IndexedList

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
sentsembs = dict()

def cossim(vec1, vec2):
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    return sim



class Markov_sif():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        maxsts = [1, 2, 3, 4, 5, max([len(i) for i in self.mydata.train])]
        self.sifmodel = self.siftrain()
        self.write_result(maxsts)

        
        
    def write_result(self, maxsts):
        for maxst in maxsts:
            self.maxst = maxst
            self.models = [Chain_withid(self.mydata.train, i).model for i in range(0, self.maxst)]
            # for mu in np.arange(1, 10):

            preds, acc = self.accu_all(self.mydata.test)
            print('{}, {}, {}'.format(seed, self.maxst, acc))
            output('{}, {}, {}'.format(seed, self.maxst, acc), filename=filename,
                   func='write')
        
    # 
    def similarity_score(self,idtest, ids):
        idsembed =[]
        for id in ids:
            idsembed.append(self.sifembed(self.mydata.titles_train[id]))
        score = cossim(np.mean(idsembed, axis=0), self.sifembed(self.mydata.titles_test[idtest]))
        return score/len(ids)

    def similarity_score2(self,idtest, ids):
        score = 0
        for id in ids:
            score += cossim(self.sifembed(self.mydata.titles_train[id]), self.sifembed(self.mydata.titles_test[idtest]))
        return score/len(ids)
    
    def predict(self, lis, id):
        history = tuple(lis)
        order = (self.maxst-1) if len(lis) > (self.maxst-1) else len(lis)
        if history in self.models[order].keys():
            normp = sum([self.models[order][history][k][0] for k in self.models[order][history].keys()])
            # norms = sum([similarity_score(id,self.models[order][history][k][1]) for k in self.models[order][history].keys()])
            self.prediction = max(self.models[order][history].keys(), key=(lambda k:  (self.models[order][history][k][0]/normp) * self.similarity_score2(id,self.models[order][history][k][1])))
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

    def sifembed(self, text):
        sent = ' '.join(text)
        if sent not in sentsembs:
            sentsembs[sent] = self.sifmodel.infer([(text, 0)])[0]
        return sentsembs[sent]

    def siftrain(self):
        glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
        model = SIF(glove, workers=1, lang_freq="en")
        sentences = IndexedList(self.mydata.titles_train)
        model.train(sentences)
        return model



if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/res/markov_sif.csv'
    seed = int(sys.argv[1])
    # order = int(sys.argv[2])
    simdic = dict()
    print('Training with seed:{}'.format(seed), flush=True)
    Markov_sif(extracted=False)
    sys.exit()

