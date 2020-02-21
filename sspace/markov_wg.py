

from chain import Chain_withid
from numpy import average
from utils.datas import Data
from utils.util import mean_confidence_interval , ngram_simscore , output

def similarity_score (id1, ids):
    score = 0
    for id in ids:
        score += ngram_simscore(mydata.titles_test[id1], mydata.titles_train[id])
    # print(score)
    return score/len(ids)





def predict(lis, id):
    global prediction
    history = tuple(lis)
    order = (maxst-1) if len(lis) > (maxst-1) else len(lis)
    if history in models[order-1].keys():
        # print('lis', lis)
        prediction = max(models[order-1][history].keys(), key=(lambda k: models[order-1][history][k][0]*similarity_score(id,models[order-1][history][k][1])))
        return prediction
    elif len(lis) > 0:
        lis2 = lis[1:]
        predict(lis2, id)
    else:
        return -1


def accu_all(test):
    corr = 0
    total = 0
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
    for n, seed in enumerate(seeds):
        global mydata
        mydata = Data(seed, titles=True)
        prediction = 0
        global maxst
        maxst = max([len(i) for i in mydata.train])
        # maxst = 2
        global models
        models = [Chain_withid(mydata.train, i).model for i in range(1,maxst)]
        preds , acc = accu_all(mydata.test)
        accu_list.append(acc)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')


if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/akom_wig_now.txt')
