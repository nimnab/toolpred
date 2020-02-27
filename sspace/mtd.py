import nltk
import numpy as np

nltk.data.path.append("/hri/localdisk/nnabizad/nltk_data")
from utils.util import output
from utils.util import mean_confidence_interval
from utils.datas import Data
import sys


class Model():
    def __init__(self, tmat, emat, vocab, num_classes):
        self.tmat = tmat
        self.emat = emat
        self.vocab = vocab
        self.num_classes = num_classes


def prior(data, mat):
    for k in range(1, maxorder + 1):
        for seq in data:
            for t in range(k, len(seq)):
                mat[k-1,seq[t],seq[t-k]] +=1
    return mat

def initialize():
    # random initialization
    mmat = np.zeros([maxorder, vocablen, vocablen])
    # mmat = np.random.rand(maxorder, vocablen, vocablen)
    landamat = [1 / maxorder] * maxorder
    phimat = np.empty([maxorder, vocablen, vocablen])

    mmat = prior(mydata.train, mmat)

    for order in mmat:
        for next_tool in range(len(order)):
            su = sum(order[:,next_tool])
            if su != 0:
                nozeros = np.count_nonzero(order[:,next_tool])
                zeros = vocablen - nozeros
                if zeros > 0:
                    for i in range(vocablen):
                        if order[i, next_tool]==0:
                            order[i, next_tool]= ((alpha*nozeros)/zeros)/su
                        else:
                            order[i, next_tool] = (order[i, next_tool]-alpha)/su
            else:
                order[:,next_tool] = 1/vocablen
    for k in range(maxorder):
        mmat[k, 0, :] = 0
        mmat[k, :, -1] = np.nan

    return landamat, mmat, phimat

def estep():
    for k in range(maxorder):
        phimat[k] = landamat[k] * mmat[k]
    denum = np.sum(phimat, axis=0)
    res = phimat/denum
    return res


def mstep(data):
    global landamat, mmat
    landanorm = 0
    landasum = np.zeros([maxorder])
    msum = np.zeros([maxorder, vocablen, vocablen])

    for k in range(1, maxorder + 1):
        for seq in data:
            for t in range(k, len(seq)):
                landasum[k-1] += phimat[k-1, seq[t], seq[t - k]]
                msum[k-1, seq[t], seq[t - k]] += phimat[k - 1, seq[t], seq[t - k]]
        landanorm += landasum[k-1]
    landamat = landasum / landanorm

    for k in range(1, maxorder + 1):
        for i in range(1,vocablen):
            norm = sum(msum[k-1,:,i])
            mmat[k-1, :, i] = msum[k-1,:,i]/norm
        mmat[k-1, 0, :] = 0
        mmat[k-1, :, -1] = np.nan


def predict(seq):
    summ = np.zeros([vocablen])
    for t in mydata.decodedic:
        for k in range(1, min(maxorder + 1, len(seq)+1)):
            summ[t] += landamat[k-1] * mmat[k-1,t,seq[-k]]
            # summ[t] += landamat[k-1] * mmat[k-1,t,seq[-k]]
    return np.nanargmax(summ)

def test():
    total = correct = 0
    for seq in mydata.test:
        olds = [0]
        for t in seq[1:]:
            total +=1
            p = predict(olds)
            if p == t:
                correct+=1

            olds.append(t)
    return correct/total

class TestData():
    def __init__(self):
        self.train = [[0,2,3,4,5],[0,1,4,5,6]]
        self.decodedic = [0,1,2,3,4,5,6]
        self.test = [[0,1,2,4,5]]

if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/val/mtd.csv'
    seed = int(sys.argv[1])
    # seed = 15
    print('Training with seed:{}'.format(seed), flush=True)
    mydata = Data(seed, encod=True)
    vocablen = len(mydata.decodedic)
    emitr = 100
    alpha = 1e-5
    file = open(filename, 'a')
    # for maxorder in range(2,11):
    maxorder = 10
    accu_list = []
    maxlanda = []
    maxaccu = 0
    landamat, mmat, phimat = initialize()
    for it in range(emitr):
        # print('Seed:{}, Iteration:{}'.format(seed,it))
        accu = test()
        estep()
        mstep(mydata.train)
        # print('Accuracy:', accu*100)
        if accu > maxaccu:
            maxaccu = accu
            maxlanda = landamat
        if accu < maxaccu and it > 20:
            accu_list.append(maxaccu)
            break
        file.write('seed: {}, iteration: {} ,accuracy{}'.format(seed, it, accu))
        file.write('\n')
        # for order, val in enumerate(landamat):
        #     file.write('{} , {} , {}'.format(order, it, val))
        #     file.write('\n')
    file.write('seed{} ,MAX{}'.format(seed, maxaccu))
    file.write('\n')
    file.close()
