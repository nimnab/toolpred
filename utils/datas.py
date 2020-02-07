import pickle as pk
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from flair.embeddings import WordEmbeddings, Sentence
import numpy as np
import os

datapath = '/hri/localdisk/nnabizad/toolpreddata'

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
            encode[j[0]] = i + 2
    biglisencoded = []
    unk = len(encode) + 2
    end = len(encode) + 3
    for lis in biglis:
        _tmp = []
        _tmp.append(1)  # beg
        for i in lis:
            if i in encode:
                _tmp.append(encode[i])
            else:
                _tmp.append(unk)  # unk
        _tmp.append(end)  ##end
        biglisencoded.append(_tmp)


class Mydata():
    def __init__(self, dat, tar, titles=None):
        self.input = dat
        self.target = tar
        self.titles = titles


class Data:
    def __init__(self, seed, deep = False, toolemb = False, titles = False, concat = False):
        class_numbers = 752
        maximum_lengh = 147
        max_titles_len = 21
        emb_dim = 100
        biglis = load_obj(os.path.join(datapath, 'biglisencoded'))
        self.train, self.test = train_test_split(biglis, test_size=0.2, random_state=seed)
        if deep:
            if toolemb:
                tool_embeding = load_obj(os.path.join(datapath, 'tool_embedding'))
                trainx = [[tool_embeding[i] for i in man[:-1]] for man in self.train]
                testx = [[tool_embeding[i] for i in man[:-1]] for man in self.test]
                trainx_final = np.asarray([np.vstack((i, np.zeros([maximum_lengh-len(i),emb_dim]))) for i in trainx])
                testx_final = np.asarray([np.vstack((i, np.zeros([maximum_lengh-len(i),emb_dim]))) for i in testx])
            else:
                trainx = sequence.pad_sequences([i[:-1] for i in self.train], maxlen=maximum_lengh, padding='post')
                testx = sequence.pad_sequences([i[:-1] for i in self.test], maxlen=maximum_lengh, padding='post')
                trainx_final = np_utils.to_categorical(trainx, num_classes=class_numbers)
                trainx_final[:, :, 0] = 0
                testx_final = np_utils.to_categorical(testx, num_classes=class_numbers)
                testx_final[:, :, 0] = 0

            train_labs = self.lable_create(self.train)
            test_labs = self.lable_create(self.test)
            trainy = sequence.pad_sequences(train_labs, maxlen=maximum_lengh, padding='post')
            testy = sequence.pad_sequences(test_labs, maxlen=maximum_lengh, padding='post')
            trainy_onehot = np_utils.to_categorical(trainy, num_classes=class_numbers)
            trainy_onehot[:,:,0]=0
            testy_onehot = np_utils.to_categorical(testy, num_classes=class_numbers)
            testy_onehot[:,:,0]=0


        titles_test = titles_train = None
        if titles and deep:
            if os.path.isfile(os.path.join(datapath, 'myglove.npy')):
                print('loading the titles from file')
                titles_pad = np.load(os.path.join(datapath, 'myglove.npy'))
            else:
                # embedding = WordEmbeddings('glove')
                embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/glove100_word2vec1')
                # embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/myword2vec_300')
                cats = load_obj(os.path.join(datapath, 'all_cats'))
                titles = [cat[0] for cat in cats]
                titles = [Sentence(i) for i in titles]
                titles = [embedding.embed(i) for i in titles]
                titles_emb = [[np.asarray(i.embedding) for i in sent[0]] for sent in titles]
                titles_pad = np.asarray([np.vstack((i, np.zeros([max_titles_len-len(i),emb_dim]))) for i in titles_emb])
                np.save(os.path.join(datapath, 'myglove'),titles_pad)
                print('file saved')

            titles_train, titles_test = train_test_split(titles_pad, test_size=0.2, random_state=seed)
            if concat:
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
    # address = os.path.join(datapath, 'all_tools'
    # encode(address)
    mydata = Data(0, deep=True, toolemb=True, titles=True,concat=True)
