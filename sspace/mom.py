import nltk, math, collections, pdb, sys, pickle
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
        plt.plot(list(range(1,1+it)), -1*tokll, col)
    plt.ylabel('-tokLL')
    plt.xlabel('EM iterations')
    
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([0,1])
    plt.xlabel('winning probability assigments')
    plt.ylabel('words/count')
    plt.bar(list(win_pr.keys()), list(win_pr.values()), width=0.01)
    plt.show()


def initialize(V, m):
    # random initialization
    global mmat, landamat, phimat
    mmat = np.random.rand(m, V, V)
    landamat = np.random.rand(m, V)
    phimat = np.random.rand(m, maxt)
    er, ec = np.shape(mmat[0])
    tr, tc = np.shape(landamat)

    for col in range(tc):
        su = sum(landamat[:, col])
        landamat[:, col] /= su

    for raw in mmat:
        for col in range(ec):
            su = sum(raw[:, col])
            raw[:, col] /= su

    checkemat = np.around(sum(mmat), 10) == np.ones((m,))
    assert np.all(checkemat), 'parameters of emission matrix(V x C) no well-formed'

    checktmat = np.around(sum(landamat), 10) == np.ones((V,))
    assert np.all(checktmat), 'parameters of transition matrix(C x V) not well-formed'

    return landamat, mmat


def numerator(k,t, seq):
    mult = 1
    for j in range(1,k-1):
        mult*= (1 - landamat[j][seq[t - j]])
    return (landamat[k][seq[t - k]] * mmat[seq[t - k]][seq[t]]) * mult


def posterior(k,t, seq):
    return numerator(k,t, seq)/sum([numerator(i,t, seq) for i in range(m)])


def estep():
    for k in range(m):
        for t in range(maxt):
            phimat[k][t]= posterior(k,t)
    return phimat
    
def mstep(data):
    k = 1
    numsum = denumsum =  0
    for w in mydata.encodedic:
        for seq in data:
            for t in range(len(seq)):
                if (w == seq[t - k]) : numsum += posterior(k,t, seq)
                for j in range(k,m):
                    if (w == seq[t - k]): denumsum += posterior(k,t, seq)
        landamat[k][w] = numsum/denumsum


def run_em(data, emitr):
    #EM algorithm implementation
    posterior = {}
    toklls = np.zeros((emitr, ))
    landa, mmat = initialize(len(data.encodedic), m)

    for it in range(emitr):
        print('Iteration:', it + 1,end='')
        posterior = estep(landa, mmat)
        tmat, emat = mstep(data)

    return tmat, emat, toklls

if __name__ == '__main__':
    END_TOK, START_TOK =  55, 1
    mydata = Data(0)
    vocablen = len(mydata.encodedic)
    m = 5
    maxt = max([len(a) for a in mydata.train])
    run_em(mydata.train, m)
    # test()