from utils.datas import Data
from utils.util import mean_confidence_interval, upload
from keras.models import Sequential
from keras.layers import TimeDistributed, Input, LSTM, Dense, Masking, Lambda, Embedding, concatenate, RepeatVector, GRU
from sklearn.utils import class_weight
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from utils.util import output
from keras.models import Model
from keras import backend as K
import os
# os.putenv('CUDA_VISIBLE_DEVICES','3')



def evaluate(mydata, preds):
    preds *= mydata.tool_mask
    lens = np.sum(np.sign(np.max(np.abs(mydata.dtest.target), 2)), 1)
    corrects = np.sum(np.equal(np.argmax(preds, 2), np.argmax(mydata.dtest.target, 2)), 1)
    return np.average(corrects / lens)


def keras_pred(mydata, modelname, seed):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtrain.input)

    main_input = Input(shape=(seqlength, tool_number), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, tool_number), name='seq_masked')(main_input)
    lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)

    _, titlelength, features = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlelength, features), dtype='float32')

    '''
    title_masked = Masking(mask_value=0, input_shape=(titlelength,), name='title_masked')(title_embeded)
    title_out = GRU(100, return_sequences=False)(title_masked)
    title_out = RepeatVector(seqlength)(title_out)
    out = concatenate([lstm_out, title_out])
    '''

    # '''
    sum_titles = Lambda(lambda x: K.sum(x, axis=1))(title_input)

    title_out = Dense(dens1_size, activation='relu')(sum_titles)
    title_out = RepeatVector(seqlength)(title_out)
    out = concatenate([lstm_out, title_out])
    # '''

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    # densout = Dropout(0.2)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs= [main_input ,title_input] , outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dtest.input,mydata.dtest.titles], mydata.dtest.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return 0


def write_result(hidden_size,dens1_size, dens2_size):
    filename = '/home/nnabizad/code/toolpred/sspace/res/realsum_{}_{}_{}.txt'.format(hidden_size, dens1_size, dens2_size)
    modelname = '/hri/localdisk/nnabizad/models/realsum_h{}_d{}_d{}'.format(hidden_size, dens1_size, dens2_size)+ '_s{}'
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    global mydata
    for seed in seeds:
        mydata = Data(seed, deep=True, title=True)
        if not os.path.isfile(modelname):
            keras_pred(mydata, modelname, seed)
        saved_model = load_model(modelname.format(seed))
        accu = saved_model.evaluate([mydata.dtest.input, mydata.dtest.titles], mydata.dtest.target)[1]
        print(accu)
        accu_list.append(accu)
    output(mean_confidence_interval(accu_list),filename=filename, func='write')
    output(accu_list,filename=filename, func='write')
    return filename

if __name__ == '__main__':
    hidden_size = 500
    dens1_size = 100
    dens2_size = 1000
    file = write_result(hidden_size,dens1_size, dens2_size)
    upload(file)
