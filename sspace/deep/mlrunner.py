from pandas import DataFrame

from sspace.deep.mlmodels import *
from utils.datas import Data
from utils.util import mean_confidence_interval
from utils.util import output
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
models = [lstm_pred, lstm_sum, lstm_gru, lstm_sif, lstm_sum_zeroh, lstm_contcat]


def write_result(hidden_size, dens1_size, dens2_size):
    model = models[modelindex]
    filename = '/home/nnabizad/code/toolpred/res/{}_{}_{}_{}.txt'.format(model.__name__, hidden_size,
                                                                                dens1_size, dens2_size)
    modelname = '/hri/localdisk/nnabizad/models/mac/{}_h{}_d{}_d{}'.format(model.__name__, hidden_size, dens1_size,
                                                                           dens2_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]
    accu_list = []
    global mydata
    for seed in seeds:
        mydata = Data(seed, deep=True, title=True, multilable=True, sif=(modelindex == 3))
        inputs = [mydata.dtest.input, [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.titles]]
        trained, history = model(mydata, modelname, seed, hidden_size, dens1_size, dens2_size)
        # saved_model = load_model(modelname.format(seed))
        # accu = trained.evaluate(inputs[modelindex], mydata.dtest.target)[1]
        predictions = trained.predict([inputs[modelindex]])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for val in thresholds:
            pred = predictions.copy()

            pred[pred >= val] = 1
            pred[pred < val] = 0

            precision = precision_score(mydata.dtest.target, pred, average='micro')
            recall = recall_score(mydata.dtest.target, pred, average='micro')
            f1 = f1_score(mydata.dtest.target, pred, average='micro')
            print("Micro-average quality numbers")
            print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    #     print(accu)
    #     accu_list.append(accu)
    #     DataFrame(history.history).to_csv(
    #         '/home/nnabizad/code/toolpred/res/deeplog/{}_seed{}_h1{}_h2{}_h3{}.csv'.format(
    #             model.__name__, seed, hidden_size, dens1_size, dens2_size))
    # output(mean_confidence_interval(accu_list), filename=filename, func='write')
    # output(accu_list, filename=filename, func='write')
    return filename


if __name__ == '__main__':
    hidden_size = 200
    dens1_size = 50
    dens2_size = 500
    modelindex = 0
    file = write_result(hidden_size, dens1_size, dens2_size)
    # upload(file)
