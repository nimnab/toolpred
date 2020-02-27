from .chain import Chain
from utils.datas import Data
from utils.util import output
import sys

def predict(lis):
    global prediction
    tmptuple = tuple(lis)
    order = (maxst - 1) if len(lis) > (maxst - 1) else len(lis)
    if tmptuple in models[order].keys():
        prediction = max(models[order][tmptuple].keys(), key=(lambda k: models[order][tmptuple][k]))
        return prediction
    else:
        if len(lis) > 0:
            lis = lis[1:]
            # print(lis)
            predict(lis)
        else:
            prediction = -1
            return prediction

# def accu_unigram(lis):
#     tot = corr = 0
#     for l in lis:
#         for i in l[1:]:
#             tot +=1
#             if i == 2:
#                 corr +=1
#     print(corr/tot)
#     return None, corr/tot

def accu_all(test):
    corr = 0
    total = 0
    preds = []
    for manual in test:
        tmppred = []
        oldtool = [1]
        for tool in manual[1:]:
            total += 1
            predict(oldtool)
            if prediction == tool:
                corr +=1
            # print('predicted {0}, tool {1}:'.format(predict(oldtool) , tool))
            oldtool.append(tool)
            tmppred.append(prediction)
        preds.append(tmppred)
    return preds, (corr)/(total)

def average_len(l):
  return int(sum(map(len, l))/len(l))+1

def write_result(filename):
    # seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    # for n, seed in enumerate(seeds):
    global mydata
    mydata = Data(seed, title=False)
    global maxst
    maxsts = [1,2,3,4,5,max([len(i) for i in mydata.train])]
    global models
    for maxst in maxsts:
        models = [Chain(mydata.train, i).model for i in range(0, maxst)]
        preds, acc = accu_all(mydata.test)
        print('{}, {}, {}'.format(seed, maxst,  acc))
        output('{}, {}, {}'.format(seed, maxst, acc), filename=filename, func='write')
        del models
        # output(accu_list,filename=filename, func='write')


if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/val/markov.csv'
    seed = int(sys.argv[1])
    print('Training with seed:{}'.format(seed), flush=True)
    write_result(filename)
    sys.exit()