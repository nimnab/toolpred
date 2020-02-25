from .chain import Chain
from utils.datas import Data
from utils.util import output
from utils.util import mean_confidence_interval


def predict(lis):
    global prediction
    tmptuple = tuple(lis)
    lenlis = (maxst-1) if len(lis) > (maxst-1) else len(lis)
    if tmptuple in models[lenlis-1].keys():
        prediction = max(models[lenlis-1][tmptuple].keys(), key=(lambda k: models[lenlis-1][tmptuple][k]))
        return prediction
    else:
        if len(lis) > 0:
            lis = lis[1:]
            # print(lis)
            predict(lis)
        else:
            prediction = -1
            return prediction

def accu_unigram(lis):
    tot = corr = 0
    for l in lis:
        for i in l[1:]:
            tot +=1
            if i == 2:
                corr +=1
    print(corr/tot)
    return None, corr/tot

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
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for n, seed in enumerate(seeds):
        mydata = Data(seed)
        prediction = 0
        global maxst
        # maxst = max([len(i) for i in mydata.train])
        maxst = 4
        global models
        models = [Chain(mydata.train, i).model for i in range(1,maxst)]
        preds , acc = accu_all(mydata.test)
        accu_list.append(acc)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')



if __name__ == '__main__':
    write_result('/home/nnabizad/code/toolpred/sspace/res/mac/4gram.txt')
    # mydata = Data(0)
    # accu_unigram(mydata.test)