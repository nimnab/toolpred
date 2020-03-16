import os
import pickle as pk
import re
from collections import defaultdict, Counter

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from flair.embeddings import WordEmbeddings,DocumentPoolEmbeddings
from flair.data import Sentence

from fse.models import SIF
from fse import IndexedList

# datapath = '/home/nnabizad/code/toolpred/data/yammly/mlyam_'
# bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/yammly/yammly_'
datapath = '/home/nnabizad/code/toolpred/data/mac/mac_tools'
bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/mac/mlmac_'
cleaner = re.compile(r'[^\'\w+\.]')
min_freq = 1

seeds = [5, 896783, 21, 322, 45234]
# glove_embedding = WordEmbeddings('glove')
glove_embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/glove100_word2vec1')
glovedim = 100
document_embeddings = DocumentPoolEmbeddings([glove_embedding],
                                             pooling='max')

def save_obj(obj, name):
    """
    Saving the pickle object
    """
    with open(name + '.pkl', 'wb') as file:
        pk.dump(obj, file, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    loading the pickle object
    """
    with open(name + '.pkl', 'rb') as file:
        return pk.load(file)

def cleantext(txt):
    txt = cleaner.sub(' ', txt)
    return txt.strip().lower()

class Mydata():
    def __init__(self, dat, tar, titles=None):
        self.input = dat
        self.target = tar
        self.titles = titles




class Data:
    def __init__(self, seed, title=False):
        train_ratio = 0.7
        validation_ratio = 0.1
        test_ratio = 0.2
        biglist = load_obj(datapath)
        self.encoder(biglist, usew2v=False)

    def encoder(self, biglist, usew2v):
        self.create_encoddic(biglist)
        self.dim = glovedim if usew2v else len(self.encoddic)
        maxlen = max([len(a) for a in biglist]) + 2
        self.encodedmanuals = np.empty((0, maxlen, self.dim))
        self.encodedlabels = np.empty((0, maxlen, self.dim))

        for manual in biglist:
            xvectors = np.zeros((maxlen, self.dim))
            yvectors = np.zeros((maxlen, self.dim))
            xvectors[0] = self.vectorize(['start'], usew2v)
            for ind,step in enumerate(manual):
                xvectors[ind+1] = self.vectorize(step, usew2v)
                yvectors[ind] = xvectors[ind+1]
            yvectors[len(manual)] = self.vectorize(['end'], usew2v)
            self.encodedmanuals = np.append(self.encodedmanuals, [xvectors], axis=0)
            self.encodedlabels = np.append(self.encodedlabels, [yvectors], axis=0)
        print()

    def encode(self, word):
        if word in self.encoddic: return self.encoddic[word]
        else: return self.encoddic['unknown']

    def vectorize(self,tup, usew2v):
        string = ' '.join(tup)
        if usew2v:
            sentence = Sentence(string)
            document_embeddings.embed(sentence)
            return sentence.embedding
        else:
            words = string.split()
            vectors = []
            for word in words:
                _vector = np_utils.to_categorical(self.encode(word), num_classes=self.dim)
                vectors.append(_vector)
            step = np.sum(vectors, axis =0)
            return np.clip(step, 0, 1)

    def create_encoddic(self, biglist):
        counts = Counter([l for j in biglist for k in j for i in k for l in i.split()])
        self.encoddic = dict()
        i = 1
        for j in counts.items():
            if j[1] > min_freq:
                self.encoddic[j[0]] = i
                i+=1
        morethanmin = len(self.encoddic)
        self.encoddic['start'] = 0
        self.encoddic['unknown'] = morethanmin + 1
        self.encoddic['end'] = morethanmin + 2

if __name__ == '__main__':
    mydata = Data(0, title=False)
    print()
    # t = Topicmodel(0)
