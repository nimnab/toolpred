from keract import get_activations
from keras.models import load_model

from sspace.deep.mlmodels import *
from utils.data_pool import Data
from utils.util import save_obj

models = [lstm_pred, lstm_sum, lstm_gru, lstm_sif, lstm_sum_zeroh, lstm_contcat]
filepath = '/home/nnabizad/code/toolpred/ipythons/svmdata/'


def noneadd(y):
    nounset = {'screwdriver', 'screw', 'cover', 'shield', 'nut', 'drive', 'board', 'assembly', 'cable', 'bracket',
               'standoff', 'tape', 'bezel', 'connector', 'sticker', 'speaker', 'antenna', 'magnet', 'panel', 'fan' ,
               'ribbon', 'gasket', 'subwoofer'}
    if y in nounset:
        return 'None ' + y
    else:
        return y


def save_data(hidden_size, gru_size, dens_size):
    data = 'macparts'

    modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(data, hidden_size, dens_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]
    for seed in seeds[0:1]:
        saved_model = load_model(modelname.format(seed))
        mydata = Data(15, usew2v=False, title=False)
        predictions = saved_model.predict(mydata.dtrain.input)
        print(saved_model.layers)
        _, _, featurelen = np.shape(predictions)
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.train)):
            for j in range(len(mydata.train[i])):
                Xs = np.append(Xs, [predictions[i][j]], axis=0)
                Ys.append([noneadd(y) for y in mydata.train[i][j]])
        np.save(filepath + '{}_xtrain'.format(data), Xs)
        save_obj(Ys, filepath + '{}_ytrain'.format(data))
        predictions = saved_model.predict(mydata.dtest.input)
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.test)):
            for j in range(len(mydata.test[i])):
                Xs = np.append(Xs, [predictions[i][j]], axis=0)
                Ys.append([noneadd(y) for y in mydata.test[i][j]])
        np.save(filepath + '{}_xtest'.format(data), Xs)
        save_obj(Ys, filepath + '{}_ytest'.format(data))
    return 0


def save_layer(layer, hidden_size, gru_size, dens_size):
    data = 'macparts'
    modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(data, hidden_size, dens_size) + '_s{}'
    seeds = [15, 896783, 9, 12, 45234]
    for seed in seeds[0:1]:
        saved_model = load_model(modelname.format(seed))
        mydata = Data(15, usew2v=False, title=False)
        print(saved_model.layers)
        # predictions = saved_model.predict(mydata.dtrain.input)
        predictions = get_activations(saved_model, mydata.dtrain.input, layer_name=layer, nodes_to_evaluate=None,
                                      output_format='simple',
                                      auto_compile=True)[layer]

        _, _, featurelen = np.shape(predictions)
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.train)):
            for j in range(len(mydata.train[i])):
                Xs = np.append(Xs, [predictions[i][j]], axis=0)
                Ys.append([noneadd(y) for y in mydata.train[i][j]])
        np.save(filepath + '{}_xtrain_{}'.format(data, layer), Xs)
        save_obj(Ys, filepath + '{}_ytrain_{}'.format(data, layer))
        predictions = get_activations(saved_model, mydata.dtest.input, layer_name=layer, nodes_to_evaluate=None,
                                      output_format='simple',
                                      auto_compile=True)[layer]
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.test)):
            for j in range(len(mydata.test[i])):
                Xs = np.append(Xs, [predictions[i][j]], axis=0)
                Ys.append([noneadd(y) for y in mydata.test[i][j]])
        np.save(filepath + '{}_xtest_{}'.format(data, layer), Xs)
        save_obj(Ys, filepath + '{}_ytest_{}'.format(data, layer))
    return 0


if __name__ == '__main__':
    hidden_size = 256
    dens_size = 256
    layers = ['masking_1', 'gru_1', 'time_distributed_1', 'dropout_1', 'dense_2']
    gru_size = 128
    modelindex = 0
    file = save_layer(layers[2], hidden_size, gru_size, dens_size)
