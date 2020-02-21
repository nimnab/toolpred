import nltk
import numpy as np

nltk.data.path.append("/hri/localdisk/nnabizad/nltk_data")
import matplotlib.pyplot as plt
from utils.datas import Data


class Model():
    def __init__(self, tmat, emat, vocab, num_classes):
        self.tmat = tmat
        self.emat = emat
        self.vocab = vocab
        self.num_classes = num_classes


def plot_stats(toklls, win_pr):
    plt.figure()
    for tokll, col in toklls:
        plt.plot(list(range(1, 1 + emitr)), -1 * tokll, col)
    plt.ylabel('-tokLL')
    plt.xlabel('EM iterations')

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([0, 1])
    plt.xlabel('winning probability assigments')
    plt.ylabel('words/count')
    plt.bar(list(win_pr.keys()), list(win_pr.values()), width=0.01)
    plt.show()

def prior(data, mat):
    for k in range(1, maxorder + 1):
        for seq in data:
            for t in range(k, len(seq)):
                mat[k-1][seq[t]][seq[t-k]] +=1
    return mat

def initialize():
    # random initialization
    mmat = np.zeros([maxorder, vocablen, vocablen])
    # mmat = np.random.rand(maxorder, vocablen, vocablen)
    landamat = [1 / maxorder] * maxorder
    phimat = np.empty([maxorder, vocablen, vocablen])

    mmat = prior(mydata.train, mmat)

    for raw in mmat:
        for col in range(len(raw)):
            su = sum(raw[col,:])
            if su != 0:
                nozeros = np.count_nonzero(raw[col,:])
                zeros = vocablen - nozeros
                for i in range(vocablen):
                    if raw[col,i]==0:
                        raw[col,i]= (alpha*nozeros/zeros)/su
                    else:
                        raw[col,i] = (raw[col,i]-alpha)/su
            else:
                raw[col, :] = 1/vocablen

    return landamat, mmat, phimat


def posterior(k, t1, t2):
    denum = sum([landamat[i] * mmat[i][t1][t2] for i in range(maxorder)])
    return (landamat[k] * mmat[k][t1][t2]) / denum


def estep():
    for k in range(maxorder):
        for t1 in mydata.decodedic:
            for t2 in mydata.decodedic:
                phimat[k][t1][t2] = posterior(k, t1, t2)
    return phimat


def mstep(data):
    global landamat, mmat
    for k in range(1, maxorder + 1):
        landasum = landanorm = 0
        msum = np.zeros([vocablen, vocablen])
        for seq in data:
            for t in range(k, len(seq)):
                landasum += phimat[k-1][seq[t]][seq[t - k]]
                landanorm += sum([phimat[i-1][seq[t]][seq[t - i]] for i in range(1, min(maxorder + 1, len(seq)+1))])
                msum[seq[t]][seq[t - k]] += phimat[k-1][seq[t]][seq[t - k]]
        landamat[k-1] = landasum / landanorm

        for i in range(1,vocablen):
            mnorm = sum(msum[i])
            # mmat[k - 1][i] = msum[i]/mnorm
            # if mnorm != 0:
            nonzeros = np.count_nonzero(msum[i])
            zeros = vocablen - nonzeros
            for j in range(vocablen):
                    if msum[i][j] ==0:
                        val = (alpha*nonzeros)/zeros
                    else:
                        val =  msum[i][j]-alpha
                    mmat[k-1][i][j] = val / mnorm
            # else:
            #     for j in range(vocablen):
            #         mmat[k - 1][i][j] = 1/vocablen
    sumlan = sum(landamat)
    landamat = [i / sumlan for i in landamat]



def predict(seq):
    summ = np.zeros([vocablen])
    for t in mydata.decodedic:
        for k in range(1, min(maxorder + 1, len(seq)+1)):
            summ[t] += landamat[k-1] * mmat[k-1][t][seq[-k]]
    return np.nanargmax(summ)

def test():
    total = correct = 0
    for seq in mydata.test:
        olds = [1]
        for t in seq[1:]:
            total +=1
            p = predict(olds)
            if p == t:
                correct+=1
            # else:
            #     print(t,p)
            olds.append(t)
    return correct/total

if __name__ == '__main__':
    mydata = Data(0, encod=True)
    vocablen = len(mydata.encodedic)
    maxt = max([len(a) for a in mydata.train])
    maxorder = 10
    emitr = 100
    alpha = 1e-10

    landamat, mmat, phimat = initialize()

    for it in range(emitr):
        print('Iteration:', it + 1)
        estep()
        mstep(mydata.train)
        accu = test()
        print('Accuracy:', accu*100)