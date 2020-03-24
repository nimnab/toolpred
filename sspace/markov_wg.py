

import sys

# from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.util import ngram_simscore, output
from .chain import Chain_withid

# from fse import IndexedList

datapath = '/hri/localdisk/nnabizad/toolpreddata/'




class Markov_wg():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        maxngram = max([len(i) for i in self.mydata.titles_train] + [len(i) for i in self.mydata.titles_test])
        maxsts = [1, 2, 3, 4, 5, max([len(i) for i in self.mydata.train])]
        self.write_result(maxsts,maxngram)

        
        
    def write_result(self, maxsts, maxngram):
        for maxst in maxsts:
            self.maxst = maxst
            self.models = [Chain_withid(self.mydata.train, i).model for i in range(0, self.maxst)]
            # for mu in np.arange(1, 10):
            #     for sigma in [0.001, 0.1, 0.5 , 5, 10]:
            # self.landa = np.random.normal(mu, sigma, maxngram)
            preds, acc = self.accu_all(self.mydata.test)
            print('{}, {}, {},'.format(seed, self.maxst, acc))
            output('{}, {}, {},'.format(seed, self.maxst, acc), filename=filename,
                   func='write')

    # 
    def similarity_score(self,id1, ids):
        score = 0
        for id in ids:
            if (' ' .join(self.mydata.titles_test[id1]),' '.join(self.mydata.titles_train[id])) not in simdic:
                sim =  ngram_simscore(self.mydata.titles_test[id1], self.mydata.titles_train[id])
                simdic[(' ' .join(self.mydata.titles_test[id1]),' '.join(self.mydata.titles_train[id]))] = sim
            score += simdic[(' ' .join(self.mydata.titles_test[id1]),' '.join(self.mydata.titles_train[id]))]
        # print(score/len(ids))
        return score/len(ids)
    
    
    def predict(self, lis, id):
        history = tuple(lis)
        order = (self.maxst-1) if len(lis) > (self.maxst-1) else len(lis)
        if history in self.models[order].keys():
            normp = sum([self.models[order][history][k][0] for k in self.models[order][history].keys()])
            # norms = sum([similarity_score(id,self.models[order][history][k][1]) for k in self.models[order][history].keys()])
            self.prediction = max(self.models[order][history].keys(), key=(lambda k:  (self.models[order][history][k][0]/normp) * self.similarity_score(id,self.models[order][history][k][1])))
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

def average_len(l):
  return int(sum(map(len, [i[0] for i in l]))/len(l))+1

if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/res/markov_target.csv'
    seed = int(sys.argv[1])
    # mu = int(sys.argv[2])
    simdic = dict()
    print('Training with seed:{}'.format(seed), flush=True)
    Markov_wg(extracted=False)
    sys.exit()

