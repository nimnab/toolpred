import itertools
import numpy as np
from chain import Chain_withid
from gensim import matutils
import sys
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from fse.models import SIF
from fse import IndexedList
from utils.datas import Data, seeds
from utils.util import output
from collections import defaultdict

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
# sentence_embedding = MeanEmbedding(lang="en")
sentsembs = dict()
# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()




#
# def embed(text):
#     sent = ' '.join(text)
#     if sent not in sentsembs:
#         sentsembs[sent] = sentence_embedding(sent)
#     return sentsembs[sent]


def cossim(vec1, vec2):
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    return sim


def average_len(l):
    return int(sum(map(len, [i[0] for i in l])) / len(l)) + 1




class AGM():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        self.sifmodel = self.siftrain()
        self.maxsts = [2,3,4, 5, max([len(i) for i in self.mydata.train])]
        self.sent_classes = self.cluster(self.mydata.titles_train, class_number)
        self.unigramdic = self.unigrams()
        print('unigram dic built')
        self.write_result(self.maxsts)
        
    def write_result(self, maxsts):
        for maxst in maxsts:
            self.maxst = maxst
            print('Order: ', maxst)
            self.models = [Chain_withid(self.mydata.train, i).model for i in range(1, self.maxst)]
            preds, acc = self.accu_all(self.mydata.test)
            print('{}, {}, {}, {}'.format(seed,class_number, self.maxst, acc))
            output('{}, {}, {}, {}'.format(seed,class_number, self.maxst, acc), filename=filename, func='write')
    
    
    def predict(self, lis, id):
        # print('##########',lis)
        history = tuple(lis)
        order = (self.maxst - 1) if len(lis) > (self.maxst - 1) else len(lis)
        # probs = [dict()] * class_number
        probs = defaultdict(list)
        if order == 0:
            if (history, id) in self.unigramdic:
                return self.unigramdic[history]
            else:
                return -1
        else:
            order -=1
            if history in self.models[order].keys():
                for clas in self.sent_classes.cluster_centers_:
                    p_goal = cossim(self.sifembed(self.mydata.titles_test[id]), clas)
                    for t in self.models[order][history].keys():
                        goals = self.models[order][history][t][1]
                        summs = 0
                        for g in goals:
                            numerator = len([i for i in goals if i == g])
                            denum = len([i for i in list(
                                itertools.chain(*[self.models[order][history][i][1] for i in self.models[order][history].keys()]))
                                         if i == g])
                            summs += (numerator / denum) * cossim(self.sifembed(self.mydata.titles_train[g]), clas)
                        probs[t].append(summs * p_goal)
    
                sumvals = {i: sum(probs[i]) for i in probs.keys()}
                self.prediction = max(sumvals, key=sumvals.get)
                return True

            elif len(lis) > 0:
                lis2 = lis[1:]
                self.predict(lis2, id)
            else:
                return -1
    
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
                    corr += 1
                oldtool.append(tool)
                tmppred.append(self.prediction)
            preds.append(tmppred)
        return preds, (corr) / (total)
    
    
    def unigrams(self):
        unigramdic = dict()
        unimodel =Chain_withid(self.mydata.train, 0).model
        probs = defaultdict(list)
        for id in range(len(self.mydata.test)):
            for history in unimodel.keys():
                for clas in self.sent_classes.cluster_centers_:
                    p_goal = cossim(self.sifembed(self.mydata.titles_test[id]), clas)
                    for t in unimodel[history].keys():
                        goals = unimodel[history][t][1]
                        summs = 0
                        for g in goals:
                            numerator = len([i for i in goals if i == g])
                            denum = len([i for i in list(
                                itertools.chain(*[unimodel[history][i][1] for i in unimodel[history].keys()]))
                                         if i == g])
                            summs += (numerator / denum) * cossim(self.sifembed(self.mydata.titles_train[g]), clas)
                        probs[t].append(summs * p_goal)
                sumvals = {i: sum(probs[i]) for i in probs.keys()}
                self.prediction = max(sumvals, key=sumvals.get)
                unigramdic[(history, id)]=self.prediction
        return unigramdic

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

    def cluster(self,sents, class_number):
        X = []
        for text in sents:
            # sent = ' '.join(text)
            X.append(self.sifembed(text))
        km = KMeans(n_clusters=class_number, init='k-means++', random_state=0)
        kmeans = km.fit(X)
        return kmeans


if __name__ == '__main__':
    filename ='/home/nnabizad/code/toolpred/sspace/res/mac/val/sif_target.csv'
    seed = int(sys.argv[1])
    class_number = int(sys.argv[2])
    # seed = 15
    print('Training with seed:{}, classes {}'.format(seed, class_number), flush=True)
    AGM(extracted = True)
    sys.exit()
