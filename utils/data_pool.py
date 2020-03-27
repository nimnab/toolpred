import pickle as pk
from collections import Counter

import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

# datapath = '/home/nnabizad/code/toolpred/data/yammly/mlyam_'
# bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/yammly/yammly_'
# bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/mac/mlmac_'

tools = '/home/nnabizad/code/toolpred/data/mac/mac_tools'
# tools = '/home/nnabizad/code/toolpred/data/yam/yam_tools'
objects = '/home/nnabizad/code/toolpred/data/mac/mac_parts'
# objects = '/home/nnabizad/code/toolpred/data/yam/yam_ings'
# topred = objects

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


def onehot(x, len):
    a = np.eye(len + 1)[x]
    a[:, 0] = 0
    return a


class Mydata():
    def __init__(self, dat, tar, titles=None):
        self.input = dat
        self.target = tar
        self.titles = titles


class Data:
    def __init__(self, usew2v=False, title=False, ml_output=False, freq_output=False, obj='mac_tools'):
        test_ratio = 0.2
        if obj.endswith('tools'):
            self.biglist = load_obj(tools)
        else:
            self.biglist = load_obj(objects)
        self.manuals, self.labels = self.encoder(self.biglist, usew2v=usew2v, ml_output=ml_output,
                                                 freq_output=freq_output)

    def generate_fold(self, seed):
        self.train, self.test = train_test_split(self.biglist, test_size=0.2, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            self.manuals,
            self.labels,
            test_size=0.2,
            random_state=seed,
        )
        self.dtrain = Mydata(X_train, y_train)
        self.dtest = Mydata(X_test, y_test)

    def encoder(self, biglist, usew2v, ml_output, freq_output):
        mlb = MultiLabelBinarizer()
        enc = OneHotEncoder(handle_unknown='ignore')
        self.create_encoddic(biglist)
        self.dim = glovedim if usew2v else len(self.encoddic)
        maxlen = max([len(a) for a in biglist]) + 2
        encodedmanuals = np.empty((0, maxlen, self.dim))
        if ml_output:
            alltools = [(i,) for j in biglist for k in j for i in k] + [('END',)]
            mlb.fit(alltools)
            encodedlabels = np.empty((0, maxlen, len(mlb.classes_)))
        elif freq_output:
            maxsteplen = max([len(j) for i in biglist for j in i])
            encodedlabels = np.empty((0, maxlen, maxsteplen + 1))
        else:
            encodedlabels = np.empty((0, maxlen, self.dim))
        for manual in biglist:
            ml_y = [(0,)] * maxlen
            freq_y = [0] * maxlen
            xvectors = np.zeros((maxlen, self.dim))
            yvectors = np.zeros((maxlen, self.dim))
            xvectors[0] = self.vectorize(['START'], usew2v)
            ind = 0
            for step in manual:
                if step:
                    xvectors[ind + 1] = self.vectorize(step, usew2v)
                    yvectors[ind] = xvectors[ind + 1]
                    ml_y[ind] = tuple(step)
                    freq_y[ind] = len(step)
                    # print(step)
                    ind += 1
            yvectors[ind] = self.vectorize(['END'], usew2v)
            ml_y[ind] = ('END',)
            freq_y[ind] = 1
            encodedmanuals = np.append(encodedmanuals, [xvectors], axis=0)
            if ml_output:
                encodedlabels = np.append(encodedlabels, [mlb.transform(ml_y)], axis=0)
            elif freq_output:
                encodedlabels = np.append(encodedlabels, [onehot(freq_y, maxsteplen)], axis=0)
            else:
                encodedlabels = np.append(encodedlabels, [yvectors], axis=0)

        return encodedmanuals, encodedlabels

    def encode(self, word):
        if word in self.encoddic:
            return self.encoddic[word]
        else:
            return self.encoddic['UNK']

    def vectorize(self, tup, usew2v):
        string = ' '.join(tup)
        if usew2v:
            sentence = Sentence(string)
            document_embeddings.embed(sentence)
            return sentence.embedding.cpu().detach().numpy()
        else:
            words = string.split()
            vectors = []
            for word in words:
                _vector = np_utils.to_categorical(self.encode(word), num_classes=self.dim)
                vectors.append(_vector)
            step = np.sum(vectors, axis=0)
            return np.clip(step, 0, 1)

    def create_encoddic(self, biglist):
        counts = Counter([l for j in biglist for k in j for i in k for l in i.split()])
        self.encoddic = dict()
        i = 1
        for j in counts.items():
            if j[1] > min_freq:
                self.encoddic[j[0]] = i
                i += 1
        morethanmin = len(self.encoddic)
        self.encoddic['START'] = 0
        self.encoddic['UNK'] = morethanmin + 1
        self.encoddic['END'] = morethanmin + 2


if __name__ == '__main__':
    mydata = Data(0, title=False, usew2v=False, freq_output=True)
    print()
    # t = Topicmodel(0)
