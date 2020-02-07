from chain import Chain
from numpy import average

from utils.datas import Data
from utils.util import output
from utils.util import mean_confidence_interval
import numpy as np

def accu1(test):
    corr = 0
    total = 0
    for i, manual in enumerate(test):
        print('Manual {0}'.format(i))
        oldtool = '___BEGIN__'
        for tool in manual:
            total += 1
            if (oldtool,) in model.keys():
                prediction = max(model[(oldtool,)].keys(), key=(lambda k: model[(oldtool,)][k]))
                print('Prediction: {0}, Observation: {1}'.format(prediction, tool))
                if (prediction == tool):
                    corr+=1
            # else: print(oldtool, tool)
            oldtool = tool
    return corr , total




def predict(lis):
    global prediction
    tmptuple = tuple(lis)
    lenlis = maxst if len(lis) > maxst else len(lis)
    if tmptuple in models[lenlis-1].keys():
        prediction = max(models[lenlis-1][tmptuple].keys(), key=(lambda k: models[lenlis-1][tmptuple][k]))
        return prediction
    else:
        if len(lis) > 0:
            lis = lis[1:]
            # print(lis)
            predict(lis)
        else:
            prediction = -1
            return prediction

def accu_unigram(lis):
    tot = corr = 0
    for l in lis:
        for i in l[1:]:
            tot +=1
            if i == 2:
                corr +=1
    print(corr/tot)
    return corr/tot

def accu_all(test):
    corr = 0
    total = 0
    preds = []
    for manual in test:
        tmppred = []
        oldtool = ['___BEGIN__']
        for tool in manual:
            total += 1
            predict(oldtool)
            # print(prediction , tool)
            if prediction == tool:
                corr +=1
            # print('predicted {0}, tool {1}:'.format(predict(oldtool) , tool))
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds, (corr-1)/(total-1)

def average_len(l):
  return int(sum(map(len, l))/len(l))+1

def write_result(filename):
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for n, seed in enumerate(seeds):
        mydata = Data(seed)
        prediction = 0
        global maxst
        maxst = max([len(i) for i in mydata.train])-1
        # maxst = 1
        global models
        models = [Chain(mydata.train, i).model for i in range(1,maxst+1)]
        preds , acc = accu_all(mydata.test)
        accu_list.append(acc)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')


def write_result_multikey(filename):
    devicelist = ['Mac Laptop']
    keywords = [('2008','2009','2010','2011', '2012','2013' ) , ('2014','2015','2016','2017')]
    mydata = Data(devicelist, keywords, 0)
    print(len(mydata.train))
    global maxst
    # maxst = max([len(i) for i in mydata.train[0]])-1
    maxst = 1
    global models
    models = [Chain(mydata.train[0], i).model for i in range(1, maxst+1)]
    for _ in devicelist:
        for counter,key in enumerate(keywords[1]):
            preds , _accu = accu_all(mydata.test[counter])
            output('{}, {} \n'.format(key, _accu),
                   filename=filename +'res_yearly.txt', func='write')

def write_result_excluded(filename):
    devicelist = ['Mac Laptop']
    keywords = ['speaker', 'hard', 'display', 'battery']
    mydata = Data(devicelist, keywords, 0 , exclude_keyword=True)
    global maxst
    global models
    for counter,key in enumerate(keywords):
        maxst = max([len(i) for i in mydata.train[counter]]) - 1
        # maxst = 3
        models = [Chain(mydata.train[counter], i).model for i in range(1, maxst+1)]
        preds , _accu = accu_all(mydata.test[counter])
        # if counter == (len(devicelist) * len(keywords) - 1):
        #     with open(filename + 'leven__predictions_' + str(key), 'w') as f:
        #         for lis in preds:
        #             [f.write(str(i) + ' ') for i in lis]
        #             f.write('\n')
        output('{}, {} \n'.format(key, _accu),
               filename=filename +'excluded.txt', func='write')


if __name__ == '__main__':
    write_result('res/bi-gram.txt')
    # mydata = Data(0)
    # accu_unigram(mydata.test)
