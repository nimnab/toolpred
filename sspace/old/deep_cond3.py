import functools
import re
# import sets
import tensorflow as tf
from tensorflow.nn import rnn_cell, dynamic_rnn

from utils.datas_withid import Data

r = re.compile(r"(?:(?<=\s)|^)(?:[a-z]|\d+)", re.I)


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class VariableSequenceLabelling:

    def __init__(self, data, target, titles,  num_hidden=500, num_layers=4):
        self.data = data
        self.target = target
        self.titles = titles
        self._num_hidden = num_hidden
        self._num_thidden = 200
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def title_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.titles), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = rnn_cell.LSTMCell(self._num_hidden)
        tcell = rnn_cell.LSTMCell(self._num_thidden)

        # cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.5)
        with tf.variable_scope('lstm2'):
            titles_output,  _ = dynamic_rnn(
                tcell,
                self.titles,
                dtype=tf.float32,
                sequence_length=self.title_length
            )

        # initial_state = tf.Variable(titles_output[:, -1, :],validate_shape=True, trainable= True)


        weight0, bias0 = self._weight_and_bias((self._num_thidden), (self._num_hidden))
        title_out = tf.matmul(titles_output[:, -1, :], weight0) + bias0
        # title_out = tf.expand_dims(title_out,1)
        # title_data = tf.concat([title_out, self.data], axis=1)



        with tf.variable_scope('lstm1'):
            output, _ = dynamic_rnn(
                cell,
                self.data,
                dtype=tf.float32,
                sequence_length=self.length
            )


        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias((self._num_hidden), num_classes)

        # title_out= tf.expand_dims(title_out,1)
        # output = output * title_out


        toadd = tf.expand_dims((output[:,-1,:]+title_out) , 1)
        output = tf.concat((output[:,:-1,:],toadd), axis=1)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):

        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight, name='weight'), tf.Variable(bias, name='bias')


if __name__ == '__main__':
    import numpy as np
    devicelist = ['Mac Laptop']
    keywords = ['']
    batchsize = 10
    counter = 0
    seeds = [0, 12, 21, 32, 45, 64, 77, 98]
    for seed in seeds:
        learning_rate = 0.01
        mydata = Data(devicelist, keywords, seed , deep = True)
        print()

        tf.reset_default_graph()
        trainlen , length, tool_number = np.shape(mydata.dtrain.input)
        _, titlelen, title_number = np.shape(mydata.dtrain.titles)
        data = tf.placeholder(tf.float32, [None, length, tool_number], name='input')
        target = tf.placeholder(tf.float32, [None, length, tool_number], name='target')
        titles = tf.placeholder(tf.float32, [None, titlelen, title_number], name='titles')

        model = VariableSequenceLabelling(data, target, titles)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        old_accu = 0
        best_epoch = 0
        print('start training with lr = {0}'.format(learning_rate))
        early = 30
        for epoch in range(200):
            for n in range(int(trainlen / batchsize)):
                sess.run(model.optimize, {data: mydata.dtrain.input[n * batchsize: (n + 1) * batchsize],
                                          target: mydata.dtrain.target[n * batchsize: (n + 1) * batchsize],
                                          titles : mydata.dtrain.titles[n * batchsize: (n + 1) * batchsize]})

            error = sess.run(model.error, {data: mydata.dtest.input, target: mydata.dtest.target, titles : mydata.dtest.titles})
            new_accu = (1 - error) * 100
            print('Epoch {:2d} accuracy {:3.1f}%'.format(epoch + 1, new_accu))
            if new_accu > old_accu and epoch > 5:
                # saver.save(sess, "models/models_lstm/conditioned/Maclaptob_{0}.ckpt".format(seed), write_meta_graph=False)
                old_accu = new_accu
                best_epoch = epoch
            if epoch - best_epoch > early : break #early stopping
            if (epoch+1)% 40==0:
                learning_rate /= 2
                print('learning rate reduced to {0}'.format(learning_rate))
            if new_accu == 0 and epoch > 5:
                best_epoch = epoch
                sess.run(tf.global_variables_initializer())
                print('zero accuracy')
                learning_rate /= 2
                early = 50
                print('learning rate reduced to {0}'.format(learning_rate))