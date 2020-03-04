import sys

import numpy as np
from .chain import Chain_withid
from nltk.util import ngrams
# from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.knlm import ModifiedKneserNey
from utils.util import ngram_simscore, output
from knlm import KneserNey
datapath = '/hri/localdisk/nnabizad/toolpreddata/'


class Markov_kn():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        horder = max([len(i) for i in self.mydata.titles_train])
        # titles_sents = '. '.join([' '.join(i) for i in self.mydata.titles_train])

        maxsts = [max([len(i) for i in self.mydata.train]), 3, 2, 1]
        self.write_result(maxsts)

    def write_result(self, maxsts):
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
    def similarity_score(self, idtest, ids):
        highest_order = max([len(i) for j in ids for i in self.mydata.titles_train[j]])
        lm = KneserNey(highest_order, 4)
        for id in ids:
            lm.train(self.mydata.titles_train[id])
        lm.optimize()
        score = lm.evaluateSent(self.mydata.titles_test[idtest])
        return score

    def predict(self, lis, id):
        history = tuple(lis)
        order = (self.maxst - 1) if len(lis) > (self.maxst - 1) else len(lis)
        if history in self.models[order].keys():
            normp = sum([self.models[order][history][k][0] for k in self.models[order][history].keys()])
            # norms = sum([similarity_score(id,self.models[order][history][k][1]) for k in self.models[order][history].keys()])
            self.prediction = max(self.models[order][history].keys(), key=(
                lambda k: np.log(self.models[order][history][k][0] / normp) + self.similarity_score(id, self.models[order][history][k][1])))
            return self.prediction
        elif len(lis) > 0:
            lis2 = lis[1:]
            self.predict(lis2, id)
        else:
            self.prediction = -1
            return None

    def accu_all(self, test):
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
                    corr += 1
                oldtool.append(tool)
                tmppred.append(self.prediction)
            preds.append(tmppred)
        return preds, (corr) / (total)


def average_len(l):
    return int(sum(map(len, [i[0] for i in l])) / len(l)) + 1


if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/res/markov_target_kn.csv'
    seed = int(sys.argv[1])
    # mu = int(sys.argv[2])
    simdic = dict()
    print('Training with seed:{}'.format(seed), flush=True)
    Markov_kn(extracted=False)
    sys.exit()
