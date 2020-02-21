from utils.datas import Data
# from utils.datas import Data
from keras.models import Sequential
from keras.layers import Dense, Masking, Dropout
from keras.layers import LSTM, TimeDistributed
from sklearn.utils import class_weight
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from utils.util import output
import os
from utils.util import mean_confidence_interval

os.putenv('CUDA_VISIBLE_DEVICES','3')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def evaluate(mydata, preds, counter):
    preds *= mydata.tool_mask[counter]
    lens = np.sum(np.sign(np.max(np.abs(mydata.dtest.target), 2)), 1)
    corrects = np.sum(np.equal(np.argmax(preds ,2),np.argmax(mydata.dtest.target ,2)),1)
    return np.average(corrects/lens)




def keras_pred(mydata, modelname, seed):
    # for seed in seeds:
    trainlen, seqlength, tool_number = np.shape(mydata.dtrain.input)
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(seqlength, tool_number)))
    model.add(LSTM(hidden_size, return_sequences=True))

    model.add(TimeDistributed(Dense(dense_size, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(tool_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(mydata.dtrain.input, mydata.dtrain.target,
              validation_data=(mydata.dtest.input,mydata.dtest.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    # np.concatenate(mydata.dtest.input, axis=0)
    return 0

def write_result(hidden_size,dense_size):

    filename = '/home/nnabizad/code/toolpred/sspace/res/res_lstm_{}_{}.txt'.format(hidden_size,dense_size)
    modelname = '/hri/localdisk/nnabizad/models/keras_{}_{}'.format(hidden_size,dense_size) + '_{}'

    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for i, seed in enumerate(seeds):
        mydata = Data(seed, deep=True)
        if not os.path.isfile(modelname.format(seed)):
            print("loading existing model")
            keras_pred(mydata, modelname, seed)
        saved_model = load_model(modelname.format(seed))

        accu = saved_model.evaluate(mydata.dtest.input, mydata.dtest.target)[1]
        print(accu)
        accu_list.append(accu)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')



if __name__ == '__main__':
    hidden_size = 512
    dense_size = 1024
    write_result(hidden_size,dense_size)
    # write_result_multikey()
