from sspace.deep.mlmodels import *
from utils.data_pool import Data
from pandas import DataFrame

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
    data = 'macparts'
    model = models[modelindex]
    filename = '/home/nnabizad/code/toolpred/res/{}.txt'.format(data)
    modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(data, hidden_size,dens_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]

    for seed in seeds[0:1]:
        inputs = [mydata.dtest.input, [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.titles]]
        trained, history = model(mydata, modelname, seed, hidden_size, gru_size, dens_size)
        # saved_model = load_model(modelname.format(seed))
        # accu = trained.evaluate(inputs[modelindex], mydata.dtest.target)[1]
        DataFrame(history.history).to_csv(
            '/home/nnabizad/code/toolpred/res/logs/{}_lstm{}_dense{}.csv'.format(data, hidden_size, dens_size))
        predictions = trained.predict(inputs[modelindex])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        with open(filename, 'a') as file:
            file.write('lstm size: {}, dense size: {}\n'.format(hidden_size, dens_size))
            for val in thresholds:
                precision = []
                recall = []
                f1 = []
                pred = predictions.copy()

                pred[pred >= val] = 1
                pred[pred < val] = 0

                for man in range(len(pred)):
                    for step in range(len(pred[man])):
                        if np.sum(mydata.dtest.target[man][step]) !=0:
                            per,rec,f_1 = per_recal(mydata.dtest.target[man][step], pred[man][step])
                            precision.append(per)
                            recall.append(rec)
                            f1.append(f_1)
                file.write("Val: {}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}\n".format(val, np.mean(precision), np.mean(recall), np.mean(f1)))
            file.write('------------\n')
    return 0


if __name__ == '__main__':
    hidden_sizes = [256 ,128, 64]
    dens_sizes = [512, 256, 128, 64]

    gru_size = 128
    modelindex = 0
    mydata = Data(15, usew2v=False, title=False)
    # for hidden_size in hidden_sizes:
    #     for dens_size in dens_sizes:
    file = write_result(mydata, 256, gru_size, 256)
