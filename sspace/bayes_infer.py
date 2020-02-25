

from .chain import Chain_withid
# from utils.sentemb import cosin_sim , fasttext , tfidf
from utils.datas import Data
from utils.util import mean_confidence_interval , ngram_simscore , output, load_obj, save_obj, get_tuples_nosentences, goal_simscore
import numpy as np
from fse import IndexedList

datapath = '/hri/localdisk/nnabizad/toolpreddata/'

# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()

def similarity_score(goal, ids):
    score = 0
    for id in ids:
        # if (' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id])) not in simdic:
        sim =  goal_simscore(goal, mydata.titles_train[id])
            # simdic[(' ' .join(mydata.titles_test[id1]),' '.join(mydata.titles_train[id]))] = sim
        score += sim
    print(score/len(ids))
    return score/len(ids)


def predict(lis, goal):
    global prediction
    history = tuple(lis)
    # landa = 0.5
    order = (maxst-1) if len(lis) > (maxst-1) else len(lis)
    if history in models[order-1].keys():
        normp = sum([models[order-1][history][k][0] for k in models[order-1][history].keys()])
        # norms = sum([similarity_score(id,models[order-1][history][k][1]) for k in models[order-1][history].keys()])
        prediction = max(models[order-1][history].keys(), key=(lambda k:  (models[order-1][history][k][0]/normp) * similarity_score(goal,models[order-1][history][k][1])))
        return prediction
    elif len(lis) > 0:
        lis2 = lis[1:]
        predict(lis2, id)
    else:
        return -1


ngramdic = dict()
def prev_goal_ngrams(history):
    history = tuple(history)
    if history not in ngramdic:
        order = len(history)
        ngrams = []
        for t in models[order - 1][history].keys():
            for g in models[order - 1][history][t][1]:
                text = mydata.titles_train[g]
                for n in range(len(text)):
                    [ngrams.append(i) for i in get_tuples_nosentences(text, n) if len(i)>0]
        ngramdic[history] = tuple(ngrams)
    return ngramdic[history]


def accu_all(test):
    corr = total = 0
    preds = []
    for manual in test:
        # print(id)
        tmppred = []
        oldtool = [1]
        for tool in manual[1:]:
            goal = prev_goal_ngrams(oldtool)
            p1 = predict(oldtool, goal)
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
    # seeds = [0]

    accu_list = []
    for n, seed in enumerate(seeds):
        global mydata
        mydata = Data(seed, titles=True)
        prediction = 0
        global maxst
        maxngram = max([len(i) for i in mydata.titles_train])
        maxst = max([len(i) for i in mydata.train])
        # maxst = 3
        global models
        global landa
        models = [Chain_withid(mydata.train, i).model for i in range(1,maxst)]
        # for mu in np.arange(1, 11):
        #     for sigma in np.arange(0.001, 11, 0.5):
        # landa = np.random.normal(mu, sigma, maxngram)
        preds , acc = accu_all(mydata.test)
        accu_list.append(acc)
        print("accuracy {}".format( acc))
    output(mean_confidence_interval(accu_list), filename=filename, func='write')
    output(accu_list, filename=filename, func='write')


if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/bayes-akom.txt')
    # save_obj(simdic, 'fast-similarities')
