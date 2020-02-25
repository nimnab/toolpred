from nltk.util import ngrams
from utils.datas import Data
from utils.util import mean_confidence_interval, output
from .chain import Chain_withid

datapath = '/hri/localdisk/nnabizad/toolpreddata/'

# simdic = load_obj(datapath + 'ngram-similarities')
simdic = dict()
dicscores = dict()


def goal_simscore(tups, text2):
    tupleslenn = max([len(i) for i in tups])
    n = max(tupleslenn, len(text2))
    score = 0
    for i in range(n):
        if (' '.join(text2), i + 1) in dicscores:
            seq2tupes = dicscores[(' '.join(text2), i + 1)]
        else:
            seq2tupes = get_tuples_nosentences(text2, i + 1)
            dicscores[(' '.join(text2), i + 1)] = seq2tupes
        score += ((i + 1) * (len(tups & seq2tupes)))
    return score


def get_tuples_nosentences(txt, n):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    txt = [i for i in txt if len(i) > 0]
    ng = ngrams(txt, n)
    return set(ng)


def similarity_score(goal, ids):
    score = 0
    for id in ids:
        sim = goal_simscore(goal, mydata.titles_train[id])
        score += sim
    return score / len(ids)


def predict(history):
    global prediction
    order = (maxst - 1) if len(history) > (maxst - 1) else len(history)
    if history in models[order - 1].keys():
        normp = sum([models[order - 1][history][k][0] for k in models[order - 1][history].keys()])
        # norms = sum([similarity_score(id,models[order-1][history][k][1]) for k in models[order-1][history].keys()])
        prediction = max(models[order - 1][history].keys(), key=(
            lambda k: (models[order - 1][history][k][0] / normp) * similarity_score(goal,
                                                                                    models[order - 1][history][k][1])))
        return prediction
    elif len(history) > 0:
        lis2 = history[1:]
        predict(lis2)
    else:
        return -1


ngramdic = dict()


def prev_goal_ngrams(history):
    order =  (maxst - 1) if len(history) > (maxst - 1) else len(history)
    global goal
    if history in models[order - 1]:
        if history not in ngramdic:
            ngrams = set()
            for t in models[order - 1][history]:
                for g in models[order - 1][history][t][1]:
                    text = mydata.titles_train[g]
                    for n in range(len(text)):
                        [ngrams.add(i) for i in get_tuples_nosentences(text, n) if len(i) > 0]
            ngramdic[history] = ngrams
        goal = ngramdic[history]
        return ngramdic[history]

    elif len(history) > 0:
        lis = history[1:]
        prev_goal_ngrams(lis)
    else:
        print('#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')


def accu_all(test):
    corr = total = 0
    preds = []
    for manual in test:
        tmppred = []
        oldtool = (1,)
        for tool in manual[1:]:
            prev_goal_ngrams(oldtool)
            # print(goal)
            predict(oldtool)
            total += 1
            if prediction == tool:
                corr += 1
            oldtool += (tool,)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds, (corr) / (total)


def write_result(filename):
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    # seeds = [0]

    accu_list = []
    for n, seed in enumerate(seeds):
        global mydata
        mydata = Data(seed, titles=True)
        prediction = 0
        global maxst
        maxst = max([len(i) for i in mydata.train])
        # maxst = 2
        global models
        global landa
        models = [Chain_withid(mydata.train, i).model for i in range(1, maxst)]
        preds, acc = accu_all(mydata.test)
        accu_list.append(acc)
        print("accuracy {}".format(acc))
    output(mean_confidence_interval(accu_list), filename=filename, func='write')
    output(accu_list, filename=filename, func='write')


if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/bayes-akom.txt')
    # save_obj(simdic, 'fast-similarities')
