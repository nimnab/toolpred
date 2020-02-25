from utils.datas import Data
from keras.models import load_model
from utils.util import output
from utils.util import mean_confidence_interval , upload
from sspace.deep.models import *

def write_result(hidden_size,dens1_size, dens2_size):

    model = lstm_gru
    toolemb = False
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/w{}_{}_{}_{}.txt'.format(model.__name__, hidden_size, dens1_size, dens2_size)
    modelname = '/hri/localdisk/nnabizad/models/mac/w{}_h{}_d{}_d{}'.format(model.__name__, hidden_size, dens1_size, dens2_size)+ '_s{}'
    seeds = [0, 12, 21, 32, 45]
    accu_list = []
    global mydata
    for seed in seeds:
        mydata = Data(seed, deep=True, titles=True, concat=False, toolemb=toolemb)
        # if not os.path.isfile(modelname):
        model(mydata, modelname, seed, hidden_size,  dens1_size, dens2_size)
        saved_model = load_model(modelname.format(seed))
        accu = saved_model.evaluate([mydata.dtest.input, mydata.dtest.titles], mydata.dtest.target)[1]
        # accu = saved_model.evaluate([mydata.dtest.titles], mydata.dtest.target)[1]
        print(accu)
        accu_list.append(accu)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')
    return filename

if __name__ == '__main__':
    hidden_size = 1000
    dens1_size = 100
    dens2_size = 1000
    file = write_result(hidden_size,dens1_size, dens2_size)
    upload(file)