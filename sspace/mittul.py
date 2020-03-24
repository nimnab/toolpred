import sys

import numpy as np
from fse import IndexedList
from fse.models import SIF
from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans

from utils.datas import Data
from utils.util import output
from .chain import Chain_withid

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
# sentence_embedding = MeanEmbedding(lang="en")
sentsembs = dict()
# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()

def cossim(vec1, vec2):
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    return sim


def average_len(l):
    return int(sum(map(len, [i[0] for i in l])) / len(l)) + 1


class Mittul():
    def __init__(self, extracted):
        self.mydata = Data(seed, title=True, extracted=extracted)
        self.sifmodel = self.siftrain()
        self.sent_classes = self.cluster(self.mydata.titles_train, class_number)
        self.write_result()

    def write_result(self):
        model = Chain_withid(self.mydata.train, 1)
        self.bigram, self.unigrams = self.creat_table(model)
        preds, acc = self.accu_all(self.mydata.test)
        print('{}, {}, {}'.format(seed, class_number, acc))
        output('{}, {}, {}'.format(seed, class_number, acc), filename=filename, func='write')

    def predict(self, lis, id):
        sum = np.zeros([len(self.mydata.decodedic)])
        for clas in range(class_number):
            first = self.unigrams[clas][lis[0]-1]
            for i in range(len(lis)-1):
                first *= self.bigram[clas][lis[i]-1][lis[i+1]-1]
            second = cossim(self.sifembed(self.mydata.titles_test[id]), self.sent_classes.cluster_centers_[clas])
            for t in self.mydata.decodedic:
                first *= self.bigram[clas][lis[-1]-1][t-1]
                sum[t-1] += (first*second)
                # probs[t]= sum

        self.prediction = np.argmax(sum)+1
        return self.prediction

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

    def creat_table(self, model):
        bigrams = np.zeros([class_number, len(self.mydata.encodedic), len(self.mydata.encodedic)])
        unigrams = np.zeros([class_number,len(self.mydata.encodedic)])
        for t in model.titles:
            ids = model.titles[t]
            for id in ids:
                title = self.sifembed(self.mydata.titles_train[id])
                p = self.sent_classes.predict(title.reshape(1, -1))[0]
                unigrams[p][t-1] += 1
        for clas in range(class_number):
            unigrams[clas] = self.smooth(unigrams[clas])

        for t1 in model.model:
            for t2 in model.model[t1]:
                ids = model.model[t1][t2][1]
                for id in ids:
                    title = self.sifembed(self.mydata.titles_train[id])
                    p = self.sent_classes.predict(title.reshape(1, -1))[0]
                    bigrams[p, t1[0]-1, t2-1] +=1

        for clas in range(class_number):
            for t in range(len(bigrams[clas])):
                bigrams[clas][t] = self.smooth(bigrams[clas][t])

        return  bigrams, unigrams
    
    def smooth(self, lis):
        su = sum(lis)
        if su != 0:
            nonzeros = np.count_nonzero(lis)
            zeros = len(lis) - nonzeros
            if zeros > 0:
                for i in range(len(lis)):
                    if lis[i] == 0:
                        lis[i] = ((alpha * nonzeros) / zeros) / su
                    else:
                        lis[i] = (lis[i] - alpha) / su
            else:
                for i in range(len(lis)):
                    lis[i]/= su
        else:
            lis = [1 / len(lis) for _ in range(len(lis))]
        return lis

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

    def cluster(self, sents, class_number):
        X = []
        for text in sents:
            X.append(self.sifembed(text))
        km = KMeans(n_clusters=class_number, init='k-means++', random_state=0)
        kmeans = km.fit(X)
        return kmeans


if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/res/Emittul.csv'
    seed = int(sys.argv[1])
    class_number = int(sys.argv[2])
    alpha = 0.001
    print('Training with seed:{}, classes {}'.format(seed, class_number), flush=True)
    Mittul(extracted=True)
    sys.exit()
