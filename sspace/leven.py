import distance
from utils.datas import Data
from collections import defaultdict
from utils.util import output, mean_confidence_interval
import numpy as np
import random
import scipy.stats


random.seed = 3

def randinsert(lis,randomportion = 0.2):
    for _ in range(int(len(lis)*randomportion)):
        lis.insert(random.randint(0,len(lis)), random.choice(lis))
    return lis


def lengthtwo(lis):
    c = 0
    for x in lis:
        for _ in x:
            c = c + 1
    return c

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
#     return m, m-h, m+h



def predict(lis, trainlis):
    toolcount = defaultdict(float)
    ln = len(trainlis)
    for tlis in trainlis:
        dmin = 999999
        # dold = 999999
        for l2 in range(1, len(tlis)):
            dis = distance.levenshtein(tlis[:l2], lis)
            if dis < dmin:
                dmin = dis
                pred = tlis[l2]
            if dmin == 0:
                break
            if l2 > len(lis) and dis > dold: break
            dold = dis
        toolcount[pred] += np.exp(-dmin*np.sqrt(ln))
        # if dmin == 0: break
    predd = max(toolcount.keys(), key=(lambda k: toolcount[k]))
    return predd


def accu_all(test, trainlis, randomportion = 0):
    preds = []
    tot = 0
    cor = 0
    for manual in test:
        _tmppred = []
        # manual = randinsert(manual, randomportion)
        oldtool = [0]
        for tool in manual[1:]:
            tot += 1
            pred = predict(oldtool, trainlis)
            _tmppred.append(pred)
            if pred == tool:
                cor += 1
            oldtool.append(tool)
        preds.append(_tmppred)
    if tot ==0: tot += 0.0000001
    return preds, cor/tot


def average_len(l):
    return int(sum(map(len, l)) / len(l)) + 1


def write_result(filename):
    devicelist = ['MacBook Pro 15" Retina', 'MacBook Pro 15"', 'MacBook Pro ']
    keywords = ['']
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    category_numbers = len(devicelist) * len(keywords)
    accu_list = np.zeros([len(seeds), category_numbers])
    # seed = seeds[9]
    for i, seed in enumerate(seeds):
        mydata = Data(devicelist, keywords, seed, notool=notool)
        for counter in range(len(devicelist)):
            trainlis = mydata.train[counter]
            preds , _seed_accu = accu_all(mydata.test[counter], trainlis)
            accu_list[i][counter] = _seed_accu
            # with open(filename + 'preds/' + str(seed), 'w') as f:
            #     for lis in preds:
            #         [f.write(str(i) + ' ') for i in lis]
            #         f.write('\n')

    counter = 0
    for dev in devicelist:
        m,h = mean_confidence_interval(accu_list[:, counter])
        for _ in keywords:
            output(
                'Device: {}, accuracy: {}, conf: {}'.format(dev, np.mean(accu_list[:, counter]), h),
                filename=filename+'res_devs', func='write')
            output(accu_list[:, counter], filename=filename+'res_devs', func='write')
            counter += 1


def write_result_multikey(filename):
    devicelist = ['Mac Laptop']
    keywords = [('2008','2009','2010','2011', '2012','2013') , ('2014', '2015', '2016','2017')]
    mydata = Data(devicelist, keywords, 0, notool=notool, multitest=True)
    trainlis = mydata.train[0]
    for counter,key in enumerate(keywords[1]):
        preds , _accu = accu_all(mydata.test[counter],trainlis)
        # if counter == (len(devicelist) * len(keywords) - 1):
        #     with open(filename + 'leven__predictions_' + str(key), 'w') as f:
        #         for lis in preds:
        #             [f.write(str(i) + ' ') for i in lis]
        #             f.write('\n')
        output('{} , {} \n'.format(key, _accu),
               filename=filename+'yearly.txt' , func='write')

def write_result_excluded(filename):
    devicelist = ['Mac Laptop']
    keywords = ['speaker', 'hard', 'display', 'battery']
    mydata = Data(devicelist, keywords, 0, exclude_keyword=True)
    for counter,key in enumerate(keywords):
        preds , _accu = accu_all(mydata.test[counter],mydata.train[counter])
        # if counter == (len(devicelist) * len(keywords) - 1):
        #     with open(filename + 'leven__predictions_' + str(key), 'w') as f:
        #         for lis in preds:
        #             [f.write(str(i) + ' ') for i in lis]
        #             f.write('\n')
        output('{} , {} \n'.format(key, _accu),
               filename=filename , func='write')


if __name__ == '__main__':
    import sys
    notool = False
    filename = 'res/leven/'
    # seed = int(sys.argv[1])
    write_result(filename)
    write_result_multikey(filename)
