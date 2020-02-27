# from CPT import CPT
from sspace.CPT import CPT
import pickle as pk
from sklearn.model_selection import train_test_split

def predict(lis):
    global prediction
    prediction = models.predict(train, [lis], k, 1)
    print(prediction)
    return prediction

def load_obj(name):
    """
    loading the pickle object
    """
    with open(name + '.pkl', 'rb') as file:
        return pk.load(file)

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
    file = open(filename, 'a')
    global train
    mydata = load_obj('/home/nnabizad/code/toolpred/data/mac/mac_' + 'encoded_tools')
    train, test = train_test_split(mydata, test_size=0.2, random_state=seed)
    global k
    # k = [1,2,3,4,5,max([len(i) for i in mydata.train])]
    global models
    k = 1
    # for maxst in maxsts:
    models  = CPT.CPT()
    models.train(train)
    preds =  models.predict(train,test, k, 1)
    # print('{}, {}, {}'.format(seed, k,  acc))
    # file.write('{}, {}, {}'.format(seed, k, acc))
    del models
    file.close()
        # output(accu_list,filename=filename, func='write')


if __name__ == '__main__':
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/cpt.csv'
    seed = 15
    print('Training with seed:{}'.format(seed), flush=True)
    write_result(filename)