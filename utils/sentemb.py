
from fse import IndexedList
from fse.models import uSIF, SIF
from utils.datas import Data
from gensim.models.keyedvectors import KeyedVectors, FastTextKeyedVectors
from numpy import dot
from gensim import matutils
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sister




datapath = '/hri/localdisk/nnabizad/toolpreddata/'
mydata = Data(0, title=True)


# glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
# model = SIF(glove, workers=1, lang_freq="en")
# sentences = IndexedList(mydata.titles_train)
# model.train(sentences)
#
sentence_embedding = sister.MeanEmbedding(lang="en")


vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,5), min_df = 0, stop_words = 'english')
X = vectorizer.fit([' '.join(s) for s in mydata.titles_train])


def tfidf(text1, text2):
    vec1 = X.transform([' '.join(text1)])
    vec2 = X.transform([' '.join(text2)])
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2))).toarray()[0][0]
    # print("sim", sim)
    return sim


def train(model, sents):
    sentences = IndexedList(sents)
    model.train(sentences)
    model.save(datapath + 'usif-model')


def fasttext(text1, text2):
    vec1 = sentence_embedding(' '.join(text1))
    vec2 = sentence_embedding(' '.join(text2))
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    # print("sim",sim)
    return sim


def sif(text1, text2):
    vec1 = model.infer([(text1,0)])
    vec2 = model.infer([(text2,0)])
    sim = np.dot(matutils.unitvec(vec1), np.transpose(matutils.unitvec(vec2)))
    # print("sim",sim)
    return sim


if __name__ == '__main__':
    # datapath = '/hri/localdisk/nnabizad/toolpreddata/'
    # glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
    # model = uSIF(glove, workers=1, lang_freq="en")
    # model.load(datapath + 'usif-model')

    print(cosin_sim('this is a test'.split(), 'another test is here'.split()))



