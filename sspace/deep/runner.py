from utils.datas import Data
from keras.models import load_model
from utils.util import output
from utils.util import mean_confidence_interval , upload
from sspace.deep.models import *
from pandas import DataFrame

def write_result(hidden_size,dens1_size, dens2_size):

    model = lstm_sum
    filename = '/home/nnabizad/code/toolpred/sspace/res/mac/val/e{}_{}_{}_{}.txt'.format(model.__name__, hidden_size, dens1_size, dens2_size)
    modelname = '/hri/localdisk/nnabizad/models/mac/e{}_h{}_d{}_d{}'.format(model.__name__, hidden_size, dens1_size, dens2_size)+ '_s{}'
    seeds = [15 , 896783 , 9 , 12 , 45234]
    accu_list = []
    global mydata
    for seed in seeds:
        mydata = Data(seed, deep=True, title=True, concat=False, sif=False, extracted=True)
        trained, history = model(mydata, modelname, seed, hidden_size,  dens1_size, dens2_size)
        # saved_model = load_model(modelname.format(seed))
        accu = trained.evaluate([mydata.dtest.input , mydata.dtest.titles], mydata.dtest.target)[1]
        print(accu)
        accu_list.append(accu)
        DataFrame(history.history).to_csv('/home/nnabizad/code/toolpred/sspace/res/mac/val/deeplog/extracted/{}_{}.csv'.format(model.__name__, seed,))
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')
    return filename

if __name__ == '__main__':
    hidden_size = 1000
    dens1_size = 100
    dens2_size = 1000
    file = write_result(hidden_size,dens1_size, dens2_size)
    # upload(file)
