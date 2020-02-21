import numpy as np

from utils.datas import Data, Topicmodel, cleantext
from utils.util import mean_confidence_interval
from utils.util import output
from .chain import Chain






def predict(lis, title):
    global prediction
    words = [cleantext(i) for i in cleantext(title).split()]
    tmptuple = tuple(lis)
    lenlis = (maxst - 1) if len(lis) > (maxst - 1) else len(lis)
    if tmptuple in models[lenlis - 1].keys():
        # tops = sum([topics.mat[k][i] for i in words])
        models[lenlis - 1][tmptuple].pop(1, None)
        # models[lenlis - 1][tmptuple].pop(751, None)
        # del models[lenlis - 1][tmptuple][1]
        # del models[lenlis - 1][tmptuple][751]
        prediction = max(models[lenlis - 1][tmptuple].keys(), key=(
            lambda k: (models[lenlis - 1][tmptuple][k] * np.prod([topics.mat[k][i] for i in words]))))
        return prediction
    else:
        if len(lis) > 0:
            lis = lis[1:]
            # print(lis)
            predict(lis, title)
        else:
            prediction = -1
            return prediction


def accu_unigram(lis):
    tot = corr = 0
    for l, title in zip(lis ,topics.cattest):
        words = [i for i in cleantext(title).split()]
        prediction = max(topics.traincount.keys(), key=(
            lambda k: topics.traincount[k]/sumtools * np.prod([topics.mat[k][i] for i in words])))
        # print(prediction)
        for tool in l[1:]:
            tot += 1
            if tool == prediction:
                corr += 1
            else: print(tool, prediction)
    print(corr/tot)
    return None, corr / tot


def accu_all(test):
    corr = 0
    total = 0
    preds = []
    for manual, title in zip(test, topics.cattest):
        tmppred = []
        oldtool = [1]
        for tool in manual[1:]:
            total += 1
            predict(oldtool, title)
            if prediction == tool:
                corr += 1
            else: print(tool, prediction)
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    print(corr/total)
    return preds, (corr) / (total)


def average_len(l):
    return int(sum(map(len, l)) / len(l)) + 1


def write_result(filename):
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for n, seed in enumerate(seeds):
        mydata = Data(seed)
        global topics
        topics = Topicmodel(seed)
        global sumtools
        sumtools = sum(topics.traincount.values())
        global maxst
        # maxst = max([len(i) for i in mydata.train])
        maxst = 2
        global models
        models = [Chain(mydata.train, i).model for i in range(1, maxst)]
        preds, acc = accu_all(mydata.test)
        accu_list.append(acc)
    output(mean_confidence_interval(accu_list), filename=filename, func='write')
    output(accu_list, filename=filename, func='write')


if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/bigram_with.txt')
    # mydata = Data(0)
    # accu_unigram(mydata.test)
