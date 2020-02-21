
from fse import IndexedList
from fse.models import uSIF, SIF
from utils.datas import Data
from gensim.models.keyedvectors import KeyedVectors, FastTextKeyedVectors
from numpy import dot
from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity

datapath = '/hri/localdisk/nnabizad/toolpreddata/'
glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
model = uSIF(glove, workers=1, lang_freq="en")
mydata = Data(0, titles=True)
sentences = IndexedList(mydata.titles_train)
model.train(sentences)
# model.load(datapath + 'usif-model')



def train(model, sents):
    sentences = IndexedList(sents)
    model.train(sentences)
    model.save(datapath + 'usif-model')


def cosin_sim(text1, text2):
    vec1 = model.infer([(text1,0)])
    vec2 = model.infer([(text2,0)])
    return cosine_similarity(matutils.unitvec(vec1), matutils.unitvec(vec2))[0][0]


if __name__ == '__main__':
    # datapath = '/hri/localdisk/nnabizad/toolpreddata/'
    # glove = KeyedVectors.load("/hri/localdisk/nnabizad/w2v/glove100_word2vec1")
    # model = uSIF(glove, workers=1, lang_freq="en")
    # model.load(datapath + 'usif-model')

    print(cosin_sim('this is a test'.split(), 'another test is here'.split()))



