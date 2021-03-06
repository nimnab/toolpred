import pickle as pk
import sys

import numpy as np
# from nltk.util import ngrams
import scipy.stats


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)



pth = '/hri/localdisk/nnabizad/'
def output(input, filename = pth + "corpus.txt", func = print , nonewline=False):
    if func:
        file = None
        end = ''
        if func == 'write':
            file = open(filename, 'a')
            func = file.write
            end = '\n'
        elif nonewline:
            func = sys.stdout.write
        if type(input) == list or type(input)==np.ndarray :
            for l in input:
                func(str(l) + end)
        elif type(input) == dict:
            for i in input:
                func(str(input[i]) + ' : ' + str(i) + end)
        else:
            func(str(input) + end)
        if file: file.close()



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def upload(file):
    import dropbox
    # Create a dropbox object using an API v2 key
    d = dropbox.Dropbox('YKzgZf7GQKUAAAAAAAAMxT74UQMneilzVkUqpb07fdthP1Vu_rdPgzwZZ_XO_ayK')
    targetfile = '/Results/' + file.split('/')[-1]
    # open the file and upload it
    with open(file, "rb") as f:
        # upload gives you metadata about the file
        # we want to overwite any previous version of the file
        meta = d.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))

dicscores = dict()

def ngram_simscore(text1, text2, landa=None):
    n = max(len(text1), len(text2))
    score = 0
    summ = sum([(i+1) for i in range(n)])
    for i in range(n):
        if (' '.join(text1), i + 1) in dicscores:
            seq1tupes = dicscores[(' '.join(text1), i + 1)]
        else:
            seq1tupes = get_tuples_nosentences(text1, i + 1)
            dicscores[(' '.join(text1), i + 1)] = seq1tupes

        if (' '.join(text2), i + 1) in dicscores:
            seq2tupes = dicscores[(' '.join(text2), i + 1)]
        else:
            seq2tupes = get_tuples_nosentences(text2, i + 1)
            dicscores[(' '.join(text2), i + 1)] = seq2tupes
        score += (i+1)/summ*(len(seq1tupes & seq2tupes))
    return score

def goal_simscore(tups, text2):
    tupleslenn = max([len(i) for i in tups])
    n = max(tupleslenn, len(text2))
    score = 0
    for i in range(n):
        # if (' '.join(text1), i + 1) in dicscores:
        #     seq1tupes = dicscores[(' '.join(text1), i + 1)]
        # else:
        #     seq1tupes = get_tuples_nosentences(text1, i + 1)
        #     dicscores[(' '.join(text1), i + 1)] = seq1tupes

        if (' '.join(text2), i + 1) in dicscores:
            seq2tupes = dicscores[(' '.join(text2), i + 1)]
        else:
            seq2tupes = get_tuples_nosentences(text2, i + 1)
            dicscores[(' '.join(text2), i + 1)] = seq2tupes
        score += ((i+1) * (len(tups & seq2tupes)))
    return score

def get_tuples_nosentences(txt, n):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    txt = [i for i in txt if len(i)>0]
    ng = ngrams(txt, n)
    return set(ng)


