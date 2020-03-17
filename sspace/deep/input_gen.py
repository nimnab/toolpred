from sspace.deep.mlmodels import *
from utils.data_pool import Data
from pandas import DataFrame
from keras.models import load_model
from utils.util import save_obj

models = [lstm_pred, lstm_sum, lstm_gru, lstm_sif, lstm_sum_zeroh, lstm_contcat]



def per_recal(target, pred):
    tp = fp = fn = 0
    for i in range(len(target)):
        if (target[i] and pred[i]): tp+=1
        if (not target[i] and pred[i]): fp+=1
        if (target[i] and not pred[i]): fn+=1
    if tp==0:
        if fn==0 and fp==0:
            return 1,1,1
        else:
            return 0,0,0
    else:
        per = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*(per*rec)/(per+rec)
        return per,rec,f1






def write_result(mydata, hidden_size, gru_size, dens_size):
    data = 'mactools'
    filename = '/home/nnabizad/code/toolpred/{}.txt'.format(data)
    modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(data, hidden_size,dens_size) + '_s{} (copy)'
    seeds = [15, 896783, 9, 12, 45234]

    for seed in seeds[0:1]:
        saved_model = load_model(modelname.format(seed))
        mydata = Data(15, usew2v=False, title=False)
        predictions = saved_model.predict(mydata.dtrain.input)
        np.save('mactools_input', predictions)
        save_obj('mactool_dic', mydata.encoddic)

        # np.savez_compressed

    return 0


if __name__ == '__main__':
    hidden_size = 256
    dens_size = 256

    gru_size = 128
    modelindex = 0
    mydata = Data(15, usew2v=False, title=False)
    file = write_result(mydata, hidden_size, gru_size, dens_size)
