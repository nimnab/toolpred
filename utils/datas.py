import pickle as pk
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import os
from collections import defaultdict, Counter
import re

# datapath = '/hri/localdisk/nnabizad/toolpred/data/'
datapath = '/home/nnabizad/code/toolpred/data/mac/mac_'
bigdatapath = '/hri/localdisk/nnabizad/toolpreddata/mac/mac_'
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





def encode(address):
    min_tool_freq = 1
    from collections import Counter
    biglis = load_obj(address)
    flatlist = []
    for lis in biglis:
        [flatlist.append(i) for i in lis]
    counts = sorted(Counter(flatlist).items(), key=lambda x: x[1], reverse=True)
    encode = dict()
    for i, j in enumerate(counts):
        if j[1] > min_tool_freq:
            encode[j[0]] = i + 1
    biglisencoded = []
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
        dic = self.matcreat(train,cattrain)
        self.vocab = set([j for i in cats for j in cleantext(i[0]).split()])
        self.mat = self.normalize(dic)
        self.traincount = Counter([i for j in train for i in j if i !=1])
            # self.traincount[1] = 1
            # self.traincount[751] = 1


    def matcreat(self, mans, cats):
        dic = dict()
        for cat, man in zip(cats, mans):
            for tool in man:
                if tool not in (1,751):
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
            _tmp = {i:matdic[tool][i]/summ for i in matdic[tool]}
            unseen = len(self.vocab) - len(_tmp)
            toadd = len(_tmp)* alpha / unseen
            newdic[tool] = dict()
            for word in self.vocab:
                if word in _tmp:
                    newdic[tool][word] = _tmp[word]-alpha
                else:
                    newdic[tool][word] = toadd
            newdic[1] = {w:1/len(self.vocab) for w in self.vocab}
            newdic[751] = {w:1/len(self.vocab) for w in self.vocab}
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
    def __init__(self, seed, encod=False, deep = False, toolemb = False, titles = False, concat = False):
        if encod:
            biglis , self.encodedic, self.decodedic = encode(datapath +  'tools')
        else:
            biglis = load_obj(datapath + 'encoded_tools')
            self.encodedic = load_obj(datapath + 'encodedic')
            self.decodedic = load_obj(datapath + 'decodedic')
        class_numbers = max([i for x in biglis for i in x]) +1
        maximum_lengh = max([len(x) for x in biglis])
        emb_dim = 100
        re_stripper = re.compile('[^\w+\/\.-]')
        self.train, self.test = train_test_split(biglis, test_size=0.2, random_state=seed)
        titles_test = titles_train = None
        if titles:
            cats = load_obj(datapath + 'cats')
            titles = [cat[0] for cat in cats]
            titles = [re.split('[, \!?:]+', i) for i in titles]
            titles = [[re_stripper.sub('', a).lower() for a in i] for i in titles]
            self.titles_train, self.titles_test = train_test_split(titles, test_size=0.2, random_state=seed)
            if deep:
                if os.path.isfile(bigdatapath + 'myglove.npy'):
                    print('loading the titles from file')
                    titles_pad = np.load(bigdatapath +  'myglove.npy')
                else:
                    # embedding = WordEmbeddings('glove')
                    embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/glove100_word2vec1')
                    # embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/myword2vec_300')
                    stitles = [Sentence(i) for i in titles]
                    max_titles_len = max([len(i) for i in stitles])
                    titles = [embedding.embed(i) for i in stitles]
                    titles_emb = [[np.asarray(i.embedding) for i in sent[0]] for sent in titles]
                    titles_pad = np.asarray([np.vstack((i, np.zeros([max_titles_len-len(i),emb_dim]))) for i in titles_emb])
                    np.save(bigdatapath +  'myglove' ,titles_pad)
                    print('file saved')
                titles_train, titles_test = train_test_split(titles_pad, test_size=0.2, random_state=seed)

        if deep:
            if toolemb:
                tool_embeding = load_obj(bigdatapath + 'tool_embedding')
                trainx = [[tool_embeding[i] for i in man[:-1]] for man in self.train]
                testx = [[tool_embeding[i] for i in man[:-1]] for man in self.test]
                trainx_final = np.asarray(
                    [np.vstack((i, np.zeros([maximum_lengh - len(i), emb_dim]))) for i in trainx])
                testx_final = np.asarray(
                    [np.vstack((i, np.zeros([maximum_lengh - len(i), emb_dim]))) for i in testx])
            else:
                trainx = sequence.pad_sequences([i[:-1] for i in self.train], maxlen=maximum_lengh, padding='post')
                testx = sequence.pad_sequences([i[:-1] for i in self.test], maxlen=maximum_lengh, padding='post')
                trainx_final = np_utils.to_categorical(trainx, num_classes=class_numbers)
                trainx_final[:, :, 0] = 0
                # trainx_final = np.expand_dims(trainx_final, 3)
                testx_final = np_utils.to_categorical(testx, num_classes=class_numbers)
                testx_final[:, :, 0] = 0
                # testx_final = np.expand_dims(testx_final, 3)

            train_labs = self.lable_create(self.train)
            test_labs = self.lable_create(self.test)
            trainy = sequence.pad_sequences(train_labs, maxlen=maximum_lengh, padding='post')
            testy = sequence.pad_sequences(test_labs, maxlen=maximum_lengh, padding='post')
            trainy_onehot = np_utils.to_categorical(trainy, num_classes=class_numbers)
            trainy_onehot[:, :, 0] = 0
            testy_onehot = np_utils.to_categorical(testy, num_classes=class_numbers)
            testy_onehot[:, :, 0] = 0

            if concat and titles:
                ttrain = np.repeat(np.expand_dims(np.sum(titles_train, axis=1) ,1), maximum_lengh, axis =1)
                ttest = np.repeat(np.expand_dims(np.sum(titles_test, axis=1) ,1), maximum_lengh, axis =1)
                titles_train = np.dstack((trainx_final,ttrain))
                ter_lens = np.sum(np.sign(np.max(np.abs(trainy_onehot), 2)), 1)
                for i in range(len(titles_train)):titles_train[i][int(ter_lens[i]):]=0
                titles_test = np.dstack((testx_final,ttest))
                test_lens = np.sum(np.sign(np.max(np.abs(testx_final), 2)), 1)
                for i in range(len(titles_test)):titles_test[i][int(test_lens[i]):]=0


            self.dtrain = Mydata(trainx_final, trainy_onehot, titles_train)
            self.dtest = Mydata(testx_final, testy_onehot, titles_test)


    def lable_create(self, biglis):
        out = []
        for lis in biglis:
            tmp = [lis[i + 1] for i in range(len(lis) - 1)]
            out.append(tmp)
        return out

if __name__ == '__main__':
    address = datapath +  'tools'
    encode(address)


    mydata = Data(0, deep=False, toolemb=False, titles=False,concat=False)
    t = Topicmodel(0)