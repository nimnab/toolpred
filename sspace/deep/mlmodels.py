import numpy as np
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import TimeDistributed, Input, LSTM, Dense, Masking, Lambda, concatenate, RepeatVector, GRU, Dropout, SimpleRNN ,Permute
from keras.models import Sequential, Model
from sklearn.utils.class_weight import compute_class_weight

dr = 0.2

def lstm_pred(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(seqlength, featurelen)))
    model.add(LSTM(hidden_size, return_sequences=True, recurrent_dropout=dr))

    model.add(TimeDistributed(Dense(dens2_size, activation='relu')))
    model.add(Dropout(dr))
    model.add(Dense(tool_number, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_binary_crossentropy', mode='max', verbose=1,
                         save_best_only=True)

    h=model.fit(mydata.dtrain.input, mydata.dtrain.target,
              validation_data=(mydata.dval.input, mydata.dval.target),
              epochs=50, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    # np.concatenate(mydata.dtest.input, axis=0)
    return model,h


def lstm_sum(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout = dr)(man_masked)
    # lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)

    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    sum_titles = Lambda(lambda x: K.sum(x, axis=1))(title_input)

    # title_out = Dense(dens1_size, activation='relu')(sum_titles)
    title_out = RepeatVector(seqlength)(sum_titles)
    out = concatenate([lstm_out, title_out])
    # '''

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h

def lstm_gru(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    # lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)
    lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout=dr)(man_masked)

    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    # title_masked = Masking(mask_value=0, input_shape=(titlelength, titlefeaturelen), name='title_masked')(title_input)
    title_out = GRU(dens1_size, return_sequences=False)(title_input)
    title_out = RepeatVector(seqlength)(title_out)
    out = concatenate([lstm_out, title_out])

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    tool_freq = [y for x in mydata.train for y in x]

    class_weights = compute_class_weight('balanced',np.unique(tool_freq), tool_freq)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc], class_weight=class_weights)
    # make a prediction
    return model,h


def lstm_sum_zeroh(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)
    featurelen = np.shape(mydata.dtest.input[0])[1]
    
    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    # cell_states = Input(np.random.normal(size=(500)), dtype='float32')

    sum_titles = Lambda(lambda x: K.sum(x, axis=1))(title_input)
    # title_out = Dense(hidden_size, activation='relu')(sum_titles)
    inital = [sum_titles, sum_titles]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    lstm_out = LSTM(titlefeaturelen, return_sequences=True, recurrent_dropout = dr)(man_masked, initial_state=inital)

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(lstm_out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h


def lstm_gru_zeroh(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)
    featurelen = np.shape(mydata.dtest.input[0])[1]
    
    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    # title_masked = Masking(mask_value=0, input_shape=(titlelength,), name='title_masked')(title_input)
    title_out = GRU(dens1_size, return_sequences=False)(title_input)
    inital = [title_out, title_out]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked, initial_state=inital)

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(lstm_out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h


def lstm_contcat(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    _, titlelength, alltitlefeaturelen = np.shape(mydata.dtrain.titles)
    

    main_input = Input(shape=(seqlength, alltitlefeaturelen), dtype='float32')

    man_masked = Masking(mask_value=0, input_shape=(seqlength, alltitlefeaturelen), name='seq_masked')(main_input)
    # lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)
    lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout=0.1)(man_masked)

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(lstm_out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dtest.titles], mydata.dtest.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h

def lstm_wem_contcat(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    _, titlelength, alltitlefeaturelen = np.shape(mydata.dtrain.titles)
    

    main_input = Input(shape=(seqlength, alltitlefeaturelen), dtype='float32')

    man_masked = Masking(mask_value=0, input_shape=(seqlength, alltitlefeaturelen), name='seq_masked')(main_input)
    sec_input = Dense(400, activation='relu', activity_regularizer=regularizers.l1(0.001))(man_masked)
    lstm_out = LSTM(hidden_size, return_sequences=True)(sec_input)
    # lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout=0.1)(sec_input)

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(lstm_out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dtest.titles], mydata.dtest.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h

def lstm_wem_sum(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    sec_input = Dense(300, activation='relu', activity_regularizer=regularizers.l1(0.001))(man_masked)
    # lstm_out = LSTM(hidden_size, return_sequences=True, , recurrent_dropout = dr)(sec_input)
    lstm_out = LSTM(hidden_size, return_sequences=True)(sec_input)

    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    sum_titles = Lambda(lambda x: K.sum(x, axis=1))(title_input)

    # title_out = Dense(dens1_size, activation='relu')(sum_titles)
    title_out = RepeatVector(seqlength)(sum_titles)
    out = concatenate([lstm_out, title_out])
    # '''

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h

def dot_(tensors):
    tensors[1] = K.repeat(tensors[1], 147)
    tensors[1] = K.permute_dimensions(tensors[1], (0, 2, 1))
    return K.dot(tensors[0], tensors[1])


def lstm_gru_mult(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    # lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)
    lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout=dr)(man_masked)
    # print('###########', lstm_out)
    _, titlelength, titlefeaturelen = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlelength, titlefeaturelen), dtype='float32')

    # title_masked = Masking(mask_value=0, input_shape=(titlelength, titlefeaturelen), name='title_masked')(title_input)
    title_out = GRU(hidden_size, return_sequences=False)(title_input)

    # title_out = K.expand_dims(title_out, 1)
    # title_out = K.repeat(title_out, seqlength)
    title_out = RepeatVector(seqlength)(title_out)

    title_out = Permute((2, 1))(title_out)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1]))([lstm_out, title_out])
    print('!!!!!!!!!!!', out)


    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    # densout = Dropout(0.5)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h

def lstm_sif(mydata, modelname, seed, hidden_size, dens1_size, dens2_size):
    # for seed in seeds:
    _, seqlength, tool_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    main_input = Input(shape=(seqlength, featurelen), dtype='float32')
    man_masked = Masking(mask_value=0, input_shape=(seqlength, featurelen), name='seq_masked')(main_input)
    lstm_out = LSTM(hidden_size, return_sequences=True, recurrent_dropout = 0.1)(man_masked)
    # lstm_out = LSTM(hidden_size, return_sequences=True)(man_masked)

    _, titlefeaturelen = np.shape(mydata.dtrain.titles)

    title_input = Input(shape=(titlefeaturelen,), dtype='float32')

    # sum_titles = Lambda(lambda x: K.sum(x, axis=1))(title_input)

    # title_out = Dense(dens1_size, activation='relu')(sum_titles)
    title_out = RepeatVector(seqlength)(title_input)
    out = concatenate([lstm_out, title_out])
    # '''

    densout = TimeDistributed(Dense(dens2_size, activation='relu'))(out)
    densout = Dropout(dr)(densout)
    last = Dense(tool_number, activation='softmax')(densout)
    model = Model(inputs=[main_input, title_input], outputs=last)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname.format(seed), monitor='val_categorical_accuracy', mode='max',
                         verbose=1, save_best_only=True)

    h=model.fit([mydata.dtrain.input, mydata.dtrain.titles], mydata.dtrain.target,
              validation_data=([mydata.dval.input, mydata.dval.titles], mydata.dval.target),
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    return model,h
