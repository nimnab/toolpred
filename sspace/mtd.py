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
        plt.plot(list(range(1, 1 + EM_ITERATIONS)), -1 * tokll, col)
    plt.ylabel('-tokLL')
    plt.xlabel('EM iterations')

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([0, 1])
    plt.xlabel('winning probability assigments')
    plt.ylabel('words/count')
    plt.bar(list(win_pr.keys()), list(win_pr.values()), width=0.01)
    plt.show()


def initialize():
    # random initialization
    mmat = np.random.rand(maxorder, vocablen, vocablen)
    landamat = [1 / maxorder] * maxorder
    phimat = np.empty([maxorder, vocablen, vocablen])

    for raw in mmat:
        for col in range(len(raw)):
            su = sum(raw[:, col])
            raw[:, col] /= su

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
            if mnorm != 0:
                zeros = vocablen - np.count_nonzero(msum[i])
                for j in range(vocablen):
                        if msum[i][j] ==0:
                            val = alpha/zeros
                        else:
                            val =  msum[i][j]-alpha
                        mmat[k-1][i][j] = val / mnorm
            else:
                for j in range(vocablen):
                    mmat[k - 1][i][j] = 1/vocablen



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



def run_em():
    global landamat, mmat, phimat
    # EM algorithm implementation
    landamat, mmat, phimat = initialize()

    for it in range(emitr):
        print('Iteration:', it + 1)
        estep()
        mstep(mydata.train)
        accu = test()
        print('Accuracy:', accu*100)
    return 0


if __name__ == '__main__':
    mydata = Data(0, encod=True)
    vocablen = len(mydata.encodedic)
    maxorder = 10
    emitr = 100
    alpha = 1e-10
    run_em()