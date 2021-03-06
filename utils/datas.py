import os
import pickle as pk
import re
from collections import defaultdict, Counter

import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from fse import IndexedList
from fse.models import SIF
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# datapath = '/home/nnabizad/code/toolpred/data/yammly/mlyam_'
# bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/yammly/yammly_'
datapath = '/home/nnabizad/code/toolpred/data/mac/mlmac_'
bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/mac/mlmac_'
cleaner = re.compile(r'[^\'\w+\.]')

seeds = [5, 896783, 21, 322, 45234]


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


sentsembs = dict()


def sifembed(text, sifmodel):
    sent = ' '.join(text)
    if sent not in sentsembs:
        sentsembs[sent] = sifmodel.infer([(text, 0)])[0]
    return sentsembs[sent]


def siftrain(data):
    glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
    model = SIF(glove, workers=1, lang_freq="en")
    sentences = IndexedList(data)
    model.train(sentences)
    return model


def encode(address, encode=None):
    biglis = load_obj(address)
    biglisencoded = []
    if not encode:
        min_tool_freq = 1
        from collections import Counter
        flatlist = []
        for lis in biglis:
            [flatlist.append(i) for i in lis]
        counts = sorted(Counter(flatlist).items(), key=lambda x: x[1], reverse=True)
        encode = dict()
        for i, j in enumerate(counts):
            if j[1] > min_tool_freq:
                encode[j[0]] = i + 1
        morethanmin = len(encode)
        encode['start'] = 0
        encode['unknown'] = morethanmin + 1
        encode['end'] = morethanmin + 2
    decode = {v: k for k, v in encode.items()}
    for lis in biglis:
        _tmp = []
        _tmp.append(encode['start'])  # beg
        for i in lis:
            if i in encode:
                _tmp.append(encode[i])
            else:
                _tmp.append(encode['unknown'])  # unk
        _tmp.append(encode['end'])  ##end
        biglisencoded.append(_tmp)
    return biglisencoded, encode, decode


class Topicmodel:
    def __init__(self, seed):
        biglis = load_obj(datapath + 'encoded_tools')
        cats = load_obj(datapath + 'cats')
        train, test = train_test_split(biglis, test_size=0.2, random_state=seed)
        cattrain, self.cattest = train_test_split([i[0] for i in cats], test_size=0.2, random_state=seed)
        dic = self.matcreat(train, cattrain)
        self.vocab = set([j for i in cats for j in cleantext(i[0]).split()])
        self.mat = self.normalize(dic)
        self.traincount = Counter([i for j in train for i in j if i != 1])
        # self.traincount[1] = 1
        # self.traincount[751] = 1

    def matcreat(self, mans, cats):
        dic = dict()
        for cat, man in zip(cats, mans):
            for tool in man:
                if tool not in (1, 751):
                    if tool in dic:
                        for word in cleantext(cat).split(): dic[tool][word] += 1
                    else:
                        dic[tool] = defaultdict(int)
                        for word in cleantext(cat).split(): dic[tool][word] += 1
        return dic

    def matcreatbyword(self, mans, cats):
        dic = dict()
        for cat, man in zip(cats, mans):
            for tool in man:
                if tool in dic:
                    for word in cleantext(cat).split(): dic[word][tool] += 1
                else:
                    dic[cleantext(word)] = defaultdict(int)
                    for word in cleantext(cat).split(): dic[cleantext(word)][tool] += 1
        return dic

    def normalize(self, matdic):
        alpha = 0.00001
        newdic = dict()
        for tool in matdic:
            summ = sum(matdic[tool].values())
            _tmp = {i: matdic[tool][i] / summ for i in matdic[tool]}
            unseen = len(self.vocab) - len(_tmp)
            toadd = len(_tmp) * alpha / unseen
            newdic[tool] = dict()
            for word in self.vocab:
                if word in _tmp:
                    newdic[tool][word] = _tmp[word] - alpha
                else:
                    newdic[tool][word] = toadd
            newdic[1] = {w: 1 / len(self.vocab) for w in self.vocab}
            newdic[751] = {w: 1 / len(self.vocab) for w in self.vocab}
        return newdic


def cleantext(txt):
    txt = cleaner.sub(' ', txt)
    return txt.strip().lower()


class Mydata():
    def __init__(self, dat, tar, titles=None):
        self.input = dat
        self.target = tar
        self.titles = titles


class Data:
    def __init__(self, seed, encod=False, deep=False, toolemb=False, title=False, multilable=False, sif=False, extracted = False):
        train_ratio = 0.7
        validation_ratio = 0.1
        test_ratio = 0.2
        suffix = ''
        if extracted:
            suffix = '_extracted'
            goldlist = load_obj(datapath + 'encoded_tools')

        if encod:
            biglis, self.encodedic, self.decodedic = encode(datapath + 'tools'+suffix)
            goldlist, _, _ = encode(datapath + 'tools', self.encodedic)

        else:
            biglis = load_obj(datapath + 'encoded_tools'+suffix)
            self.encodedic = load_obj(datapath + 'encodedic')
            self.decodedic = load_obj(datapath + 'decodedic')

        if multilable:
            class_numbers = self.encodedic['end'][0]+1
        else:
            class_numbers = max([i for x in biglis for i in x]) + 1
        maximum_lengh = max([len(x) for x in biglis])
        emb_dim = 100
        re_stripper = re.compile('[^\w+\/\.-]')

        self.train, self.test = train_test_split(biglis, test_size=1 - train_ratio, random_state=seed)

        if extracted:
            _, self.test = train_test_split(goldlist, test_size=1 - train_ratio, random_state=seed)

        titles_test = titles_train = titles_val = None
        if title:
            cats = load_obj(datapath + 'cats')
            titles = [cat[0] for cat in cats]
            titles = [re.split('[, \!?:]+', i) for i in titles]
            titles = [[re_stripper.sub('', a).lower() for a in i] for i in titles]
            self.titles_train, self.titles_test = train_test_split(titles, test_size=1 - train_ratio, random_state=seed)

            if deep:
                if sif:
                    titles_pad = []
                    sifmodel = siftrain(titles)
                    for tit in titles:
                        titles_pad.append(sifembed(tit, sifmodel))
                    titles_pad = np.asarray(titles_pad)
                else:
                    if os.path.isfile(bigdatapath + 'myglove.npy'):
                        print('loading the titles from file')
                        titles_pad = np.load(bigdatapath + 'myglove.npy')
                    else:
                        # embedding = WordEmbeddings('glove')
                        embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/glove100_word2vec1')
                        # embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/myword2vec_300')
                        stitles = [Sentence(' '.join(i)) for i in titles]
                        max_titles_len = max([len(i) for i in stitles])
                        titles = [embedding.embed(i) for i in stitles]
                        titles_emb = [[np.asarray(i.embedding) for i in sent[0]] for sent in titles]
                        titles_pad = np.asarray(
                            [np.vstack((i, np.zeros([max_titles_len - len(i), emb_dim]))) for i in titles_emb])
                        np.save(bigdatapath + 'myglove', titles_pad)
                        print('file saved')
                titles_train, titles_test_ = train_test_split(titles_pad, test_size=1 - train_ratio, random_state=seed)
                titles_val, titles_test = train_test_split(titles_test_,
                                                           test_size=test_ratio / (test_ratio + validation_ratio),
                                                           random_state=seed)
        if deep:
            if toolemb:
                tool_embeding = load_obj(bigdatapath + 'tool_embedding')
                data = [[tool_embeding[i] for i in man[:-1]] for man in biglis]
                data_final = np.asarray(
                    [np.vstack((i, np.zeros([maximum_lengh - len(i), emb_dim]))) for i in data])

            if multilable:
                data = []
                for rec in biglis:
                    _tmprec = []
                    for step in rec:
                        tools = np.sum(np.asarray([np_utils.to_categorical(i, num_classes=class_numbers) for i in step]),
                               axis=0)
                        _tmprec.append(tools)

                    data.append(_tmprec)
                labels = self.lable_create(data)
                data = [i[:-1] for i in data]
                data_final = sequence.pad_sequences(data, maxlen=maximum_lengh, padding='post')
                labels_onehot = sequence.pad_sequences(labels, maxlen=maximum_lengh, padding='post')


            else:
                data = sequence.pad_sequences([i[:-1] for i in biglis], maxlen=maximum_lengh, padding='post')
                data_final = np_utils.to_categorical(data, num_classes=class_numbers)
                data_final[:, :, 0] = 0
                labels = self.lable_create(biglis)
                labels_padded = sequence.pad_sequences(labels, maxlen=maximum_lengh, padding='post')
                labels_onehot = np_utils.to_categorical(labels_padded, num_classes=class_numbers)
                labels_onehot[:, :, 0] = 0


            trainx_final, testx_ = train_test_split(data_final, test_size=1 - train_ratio, random_state=seed)
            valx_final, testx_final = train_test_split(testx_,
                                                       test_size=test_ratio / (test_ratio + validation_ratio),
                                                       random_state=seed)
            if extracted:
                # _, otestx = train_test_split(goldlist, test_size=1 - train_ratio, random_state=seed)
                testx = sequence.pad_sequences([i[:-1] for i in self.test], maxlen=maximum_lengh, padding='post')
                testx = np_utils.to_categorical(testx, num_classes=class_numbers)
                testx[:, :, 0] = 0
                valx_final, testx_final = train_test_split(testx,
                                                           test_size=test_ratio / (test_ratio + validation_ratio),
                                                           random_state=seed)

            trainy_final, testy = train_test_split(labels_onehot, test_size=1 - train_ratio, random_state=seed)
            valy_final ,testy_final = train_test_split(testy,
                                                       test_size=test_ratio / (test_ratio + validation_ratio),
                                                       random_state=seed)
            if extracted:
                testlabels = self.lable_create(self.test)
                test_labels_padded = sequence.pad_sequences(testlabels, maxlen=maximum_lengh, padding='post')
                test_labels_onehot = np_utils.to_categorical(test_labels_padded, num_classes=class_numbers)
                test_labels_onehot[:, :, 0] = 0
                valy_final, testy_final = train_test_split(test_labels_onehot,
                                                           test_size=test_ratio / (test_ratio + validation_ratio),
                                                           random_state=seed)

            self.dtrain = Mydata(trainx_final, trainy_final, titles_train)
            self.dtest = Mydata(testx_final, testy_final, titles_test)
            self.dval = Mydata(valx_final, valy_final, titles_val)

    def lable_create(self, biglis):
        out = []
        for lis in biglis:
            tmp = [lis[i + 1] for i in range(len(lis) - 1)]
            out.append(tmp)
        return out


if __name__ == '__main__':
    address = datapath + 'tools'
    encode(address)

    mydata = Data(0, deep=True, toolemb=False, title=True, multilable=True, encod = False)
    print()
    # t = Topicmodel(0)
