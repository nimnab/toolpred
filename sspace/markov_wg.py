

from .chain import Chain_withid
from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.util import mean_confidence_interval , ngram_simscore , output, load_obj, save_obj
import numpy as np

datapath = '/hri/localdisk/nnabizad/toolpreddata/'

# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()

def similarity_score(id1, ids):
    score = 0
    for id in ids:
        if (' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id])) not in simdic:
            sim =  tfidf(mydata.titles_test[id1], mydata.titles_train[id])
            simdic[(' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id]))] = sim
        score += simdic[(' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id]))]
    # print(score)
    return score/len(ids)


def predict(lis, id):
    global prediction
    history = tuple(lis)
    # landa = 0.5
    order = (maxst-1) if len(lis) > (maxst-1) else len(lis)
    if history in models[order-1].keys():
        normp = sum([models[order-1][history][k][0] for k in models[order-1][history].keys()])
        norms = sum([similarity_score(id,models[order-1][history][k][1]) for k in models[order-1][history].keys()])
        prediction = max(models[order-1][history].keys(), key=(lambda k: landa * np.log(models[order-1][history][k][0]/normp) + (1-landa)*np.log(similarity_score(id,models[order-1][history][k][1]))))
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
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for n, seed in enumerate(seeds[0:1]):
        global mydata
        mydata = Data(seed, titles=True)
        prediction = 0
        global maxst
        maxst = max([len(i) for i in mydata.train])
        # maxst = 2
        global models
        models = [Chain_withid(mydata.train, i).model for i in range(1,maxst)]
        global landa
        for landa in np.arange(0, 1.2, 0.2):
            preds , acc = accu_all(mydata.test)
            accu_list.append(acc)
            print("landa {}, accuracy {}".format(landa, acc))
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')


if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/akom_wig_tfidf.txt')
    # save_obj(simdic, 'fast-similarities')
