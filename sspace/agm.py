import itertools
import numpy as np
from .chain import Chain_withid
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

#
# def embed(text):
#     sent = ' '.join(text)
#     if sent not in sentsembs:
#         sentsembs[sent] = sentence_embedding(sent)
#     return sentsembs[sent]


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

    if history in models[order].keys():
        for clas in sent_classes.cluster_centers_:
            p_goal = cossim(sifembed(mydata.titles_test[id]), clas)
            for t in models[order][history].keys():
                goals = models[order][history][t][1]
                summs = 0
                for g in goals:
                    numerator = len([i for i in goals if i == g])
                    denum = len([i for i in list(
                        itertools.chain(*[models[order][history][i][1] for i in models[order][history].keys()]))
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
            prediction = predict(oldtool, id)
            total += 1
            if prediction == tool:
                corr += 1
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds, (corr) / (total)


def average_len(l):
    return int(sum(map(len, [i[0] for i in l])) / len(l)) + 1


def cluster(sents, class_number):
    X = []
    for text in sents:
        # sent = ' '.join(text)
        X.append(sifembed(text))
    km = KMeans(n_clusters=class_number, init='k-means++', random_state=0)
    kmeans = km.fit(X)
    return kmeans


def write_result(filename):

    # for n, seed in enumerate(seeds):
    global mydata
    mydata = Data(seed, title=True)
    global sifmodel
    sifmodel = siftrain()
    global sent_classes
    global maxst
    maxsts = [1,2,3,4, 5, max([len(i) for i in mydata.train])]
    global models
    # for class_number in range(10, 200, 20):
    for maxst in maxsts:
        models = [Chain_withid(mydata.train, i).model for i in range(0, maxst)]
    # for class_number in range(10, 200, 20):
        sent_classes = cluster(mydata.titles_train, class_number)
        preds, acc = accu_all(mydata.test)
        print('{}, {}, {}, {}'.format(seed,class_number, maxst, acc))
        output('{}, {}, {}, {}'.format(seed,class_number, maxst, acc), filename=filename, func='write')


if __name__ == '__main__':
    filename ='/home/nnabizad/code/toolpred/sspace/res/mac/val/sif_target.csv'
    seed = int(sys.argv[1])
    class_number = int(sys.argv[2])
    # seed = 15
    print('Training with seed:{}, classes {}'.format(seed, class_number), flush=True)
    write_result(filename)
    sys.exit()
