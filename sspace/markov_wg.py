

import sys

import numpy as np

# from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.util import ngram_simscore, output
from .chain import Chain_withid

# from fse import IndexedList

datapath = '/hri/localdisk/nnabizad/toolpreddata/'

simdic = dict()

def similarity_score(id1, ids):
    score = 0
    for id in ids:
        # if (' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id])) not in simdic:
        sim =  ngram_simscore(mydata.titles_test[id1], mydata.titles_train[id], landa=landa)
            # simdic[(' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id]))] = sim
        score += sim
    # print(score/len(ids))
    return score/len(ids)


def predict(lis, id):
    global prediction
    history = tuple(lis)
    order = (maxst-1) if len(lis) > (maxst-1) else len(lis)
    if history in models[order].keys():
        normp = sum([models[order][history][k][0] for k in models[order][history].keys()])
        # norms = sum([similarity_score(id,models[order][history][k][1]) for k in models[order][history].keys()])
        prediction = max(models[order][history].keys(), key=(lambda k:  (models[order][history][k][0]/normp) * similarity_score(id,models[order][history][k][1])))
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
                corr +=1
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds , (corr)/(total)

def average_len(l):
  return int(sum(map(len, [i[0] for i in l]))/len(l))+1


def write_result(filename):
    global mydata
    mydata = Data(seed, titles=True)
    prediction = 0
    global maxst
    maxngram = max([len(i) for i in mydata.titles_train])
    maxsts = [1,2,3, max([len(i) for i in mydata.train])]
    global models
    global landa
    for maxst in maxsts:
        models = [Chain_withid(mydata.train, i).model for i in range(0,maxst)]
        for mu in np.arange(1, 11):
            for sigma in [0.001, 0.1, 0,5, 1, 5, 10]:
                landa = np.random.normal(mu, sigma, maxngram)
                preds , acc = accu_all(mydata.test)
                print('{}, {}, {}. {}'.format(seed, mu, sigma, acc))
                output('{}, {}, {}. {}'.format(seed, mu, sigma, acc),filename=filename, func='write')

if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/ngram_target.csv'
    seed = int(sys.argv[1])
    print('Training with seed:{}'.format(seed), flush=True)
    write_result(filename)

