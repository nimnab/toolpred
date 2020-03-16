from sspace.deep.mlmodels import *
from utils.data_pool import Data

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






def write_result(hidden_size, gru_size, dens_size):
    model = models[modelindex]
    filename = '/home/nnabizad/code/toolpred/res/{}_{}_{}_{}.txt'.format(model.__name__, hidden_size,
                                                                         gru_size, dens_size)
    modelname = '/hri/localdisk/nnabizad/models/mac/{}_h{}_d{}_d{}'.format(model.__name__, hidden_size, gru_size,
                                                                           dens_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]
    accu_list = []
    global mydata
    for seed in seeds[0:1]:
        mydata = Data(seed, usew2v=False, title=False)
        inputs = [mydata.dtest.input, [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.titles]]
        trained, history = model(mydata, modelname, seed, hidden_size, gru_size, dens_size)
        # saved_model = load_model(modelname.format(seed))
        # accu = trained.evaluate(inputs[modelindex], mydata.dtest.target)[1]

        predictions = trained.predict(inputs[modelindex])
    return filename


if __name__ == '__main__':
    hidden_size = 256
    dens_size = 512
    gru_size = 128
    modelindex = 0
    file = write_result(hidden_size, gru_size, dens_size)
    # upload(file)
