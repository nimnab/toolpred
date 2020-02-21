from utils.datas_withid import Data
from utils.util import mean_confidence_interval
from keras.models import Sequential
from keras.layers import TimeDistributed, Input, LSTM, Dense, Masking, GRU, concatenate, RepeatVector, Lambda, \
    Dropout
from sklearn.utils import class_weight
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from utils.util import output
from keras.models import Model
from keras import backend as K
import os

# from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def evaluate(mydata, preds):
    preds *= mydata.tool_mask
    lens = np.sum(np.sign(np.max(np.abs(mydata.dtest.target), 2)), 1)
    corrects = np.sum(np.equal(np.argmax(preds ,2),np.argmax(mydata.dtest.target ,2)),1)
    return np.average(corrects/lens)

def keras_pred(mydata, modelname, seed):
    hidden_size = 500
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtrain.input)
    _, titlelength, features = np.shape(mydata.dtrain.titles)
    title_input = Input(shape=(titlelength, features), dtype='float32')

    # cell_states = Input(np.random.normal(size=(500)), dtype='float32')

    sum_titles = Lambda(lambda x: K.mean(x, axis=1))(title_input)
    title_out = Dense(hidden_size, activation='relu')(sum_titles)
    inital = [title_out, title_out]

    main_input = Input(shape=(seqlength, tool_number), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, tool_number), name='seq_masked')(main_input)
    lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked, initial_state=inital)



    '''
    title_masked = Masking(mask_value=0, input_shape=(titlelength, features), name='title_masked')(title_input)
    title_out = GRU(50, return_sequences=False)(title_masked)
    title_out = RepeatVector(seqlength)(title_out)
    out = concatenate([lstm_out, title_out])
    '''

    # '''

    # out = concatenate([lstm_out, title_out])
    # '''


    densout = TimeDistributed(Dense(200, activation='relu'))(lstm_out)
    densout = Dropout(0.5)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dtest.input, mydata.dtest.titles], mydata.dtest.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return 0


def write_result():
    filename = 'res/lstm/removed/stitle_wdr_zero_res.txt'
    modelname = '/hri/localdisk/nnabizad/models/rstitle_wdr_zero_model_{}.h5'
    devicelist = ['Mac Laptop']
    keywords = ['']
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    accu_list = []
    for seed in seeds:
        mydata = Data(devicelist, keywords, seed, notool=False, deep=True)
        if not os.path.isfile(modelname.format(seed)):
            keras_pred(mydata, modelname, seed)
        saved_model = load_model(modelname.format(seed))
        for counter in range(len(devicelist)):
            # preds = saved_model.predict([mydata.dtest.input,mydata.dtest.titles])
            # accu = evaluate(mydata,preds)
            accu = saved_model.evaluate([mydata.dtest.input,mydata.dtest.titles], mydata.dtest.target)[1]
            accu_list.append(accu)

    output('---------------------', filename=filename, func='write')
    _m, _h = mean_confidence_interval(accu_list)
    output('average {}, h {} \n'.format(_m, _h),
           filename=filename, func='write')
    output(accu_list, filename=filename, func='write')




if __name__ == '__main__':
    write_result()
