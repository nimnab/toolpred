from sspace.deep.mlmodels import *
from utils.datas import Data

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






def write_result(hidden_size, gru_size, dens2_size):
    model = models[modelindex]
    filename = '/home/nnabizad/code/toolpred/res/{}_{}_{}_{}.txt'.format(model.__name__, hidden_size,
                                                                         gru_size, dens2_size)
    modelname = '/hri/localdisk/nnabizad/models/mac/{}_h{}_d{}_d{}'.format(model.__name__, hidden_size, gru_size,
                                                                           dens2_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]
    accu_list = []
    global mydata
    for seed in seeds[0:1]:
        mydata = Data(seed, deep=True, title=True, multilable=True, sif=(modelindex == 3))
        inputs = [mydata.dtest.input, [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.input, mydata.dtest.titles],
                  [mydata.dtest.input, mydata.dtest.titles], [mydata.dtest.titles]]
        trained, history = model(mydata, modelname, seed, hidden_size, gru_size, dens2_size)
        # saved_model = load_model(modelname.format(seed))
        # accu = trained.evaluate(inputs[modelindex], mydata.dtest.target)[1]

        predictions = trained.predict(inputs[modelindex])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
            print("Micro-average quality numbers with val = {}".format(val))
            print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(np.mean(precision), np.mean(recall), np.mean(f1)))

    #     print(accu)
    #     accu_list.append(accu)
    #     DataFrame(history.history).to_csv(
    #         '/home/nnabizad/code/toolpred/res/deeplog/{}_seed{}_h1{}_h2{}_h3{}.csv'.format(
    #             model.__name__, seed, hidden_size, dens1_size, dens2_size))
    # output(mean_confidence_interval(accu_list), filename=filename, func='write')
    # output(accu_list, filename=filename, func='write')
    return filename


if __name__ == '__main__':
    hidden_size = 256
    gru_size = 128
    dens2_size = 256
    modelindex = 0
    file = write_result(hidden_size, gru_size, dens2_size)
    # upload(file)
