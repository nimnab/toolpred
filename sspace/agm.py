import itertools
import numpy as np
from .chain import Chain_withid
from gensim import matutils
from sister import MeanEmbedding
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from fse.models import SIF
from fse import IndexedList
from utils.datas import Data
from utils.util import output
from collections import defaultdict

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
sentence_embedding = MeanEmbedding(lang="en")
sentsembs = dict()
# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()

def sifembed(text):
    sent = ' '.join(text)
    if sent not in sentsembs:
        sentsembs[sent] = sifmodel.infer([(text,0)])[0]
    return sentsembs[sent]



def siftrain():
    glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
    model = SIF(glove, workers=1, lang_freq="en")
    sentences = IndexedList(mydata.titles_train)
    model.train(sentences)
    return model



def embed(text):
    sent = ' '.join(text)
    if sent not in sentsembs:
        sentsembs[sent] = sentence_embedding(sent)
    return sentsembs[sent]


def cossim(vec1, vec2):
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    return sim


def predict(lis, id):
    # print('##########',lis)
    global prediction
    history = tuple(lis)
    order = (maxst - 1) if len(lis) > (maxst - 1) else len(lis)
    # probs = [dict()] * class_number
    probs = defaultdict(list)

    if history in models[order - 1].keys():
        for clas in sent_classes.cluster_centers_:
            p_goal = cossim(sifembed(mydata.titles_test[id]), clas)
            for t in models[order - 1][history].keys():
                goals = models[order - 1][history][t][1]
                summs = 0
                for g in goals:
                    numerator = len([i for i in goals if i == g])
                    denum = len([i for i in list(
                        itertools.chain(*[models[order - 1][history][i][1] for i in models[order - 1][history].keys()]))
                                 if i == g])
                    # print('denum:', denum, 'num:', numerator)
                    summs += (numerator / denum) * cossim(sifembed(mydata.titles_train[g]), clas)
                probs[t].append(summs * p_goal)

        sumvals = {i: sum(probs[i]) for i in probs.keys()}
        prediction = max(sumvals, key=sumvals.get)
        return prediction

    elif len(lis) > 0:
        lis2 = lis[1:]
        predict(lis2, id)
    else:
        return -1

def accu_all(test):
    corr = total = 0
    preds = []
    for id, manual in enumerate(test):
        # print(id)
        tmppred = []
        oldtool = [1]
        for tool in manual[1:]:
            p1 = predict(oldtool, id)
            total += 1
            if prediction == tool:
                corr += 1
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds, (corr) / (total)


def average_len(l):
    return int(sum(map(len, [i[0] for i in l])) / len(l)) + 1


def cluster(sents):
    X = []
    for text in sents:
        # sent = ' '.join(text)
        X.append(sifembed(text))
    km = KMeans(n_clusters=class_number, init='k-means++', random_state=0)
    kmeans = km.fit(X)
    return kmeans


def write_result(filename):
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    for n, seed in enumerate(seeds):
        global mydata
        mydata = Data(seed, titles=True)
        global sifmodel
        sifmodel = siftrain()
        global sent_classes
        sent_classes = cluster(mydata.titles_train)
        global maxst
        maxsts = [1,2,3,4,max([len(i) for i in mydata.train])]
        # maxsts = [1]
        global models
        for maxst in maxsts:
            models = [Chain_withid(mydata.train, i).model for i in range(0, maxst)]
            preds, acc = accu_all(mydata.test)
            print('{}, {}, {}, {}'.format(maxst, seed, class_number, acc))
            output('{}, {}, {}, {}'.format(maxst, seed, class_number, acc), filename=filename, func='write')
        # output(accu_list,filename=filename, func='write')


if __name__ == '__main__':
    filename ='/home/nnabizad/code/toolpred/sspace/res/mac/bigram-sif.txt'
    output('order, seed, class_number, acc', filename=filename, func='write')
    for class_number in range(10,100,10):
        write_result(filename)
