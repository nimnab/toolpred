import functools
import numpy as np
import re
# import sets
import tensorflow as tf
from tensorflow.nn import rnn_cell, dynamic_rnn

from utils.datas import Data
from utils.util import output

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

    def __init__(self, data, target, num_hidden=500, num_layers=4):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
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
    def prediction(self):
        # Recurrent network.
        output, _ = dynamic_rnn(
            rnn_cell.LSTMCell(self._num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
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
        learning_rate = 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
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
        return tf.Variable(weight), tf.Variable(bias)


if __name__ == '__main__':
    # import sys
    devicelist = ['MacBook Pro 15" Retina', 'MacBook Pro 15"', 'MacBook Pro ', 'Mac Laptop']
    keywords = ['speaker', 'battery', '']
    seeds = [0, 12, 21, 32, 45, 64, 77, 98]

    counter = 0
    for device in devicelist:
        for keyword in keywords:
            _accu = []
            for seed in seeds:
                mydata = Data(devicelist, keywords, seed, deep=True)
                tf.reset_default_graph()
                sess = tf.Session()
                _accu_seed = []
                batchsize = 10
                trainlen , length, tool_number = np.shape(mydata.dtrain.input[counter])
                data = tf.placeholder(tf.float32, [None, length, tool_number], name='input')
                target = tf.placeholder(tf.float32, [None, length, tool_number], name='target')
                model = VariableSequenceLabelling(data, target)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                for epoch in range(30):
                    for n in range(int(trainlen/batchsize)):
                        sess.run(model.optimize, {data: mydata.dtrain.input[counter][n * batchsize: (n + 1) * batchsize], target: mydata.dtrain.target[counter][n * batchsize: (n + 1) * batchsize]})
                    error = sess.run(model.error, {data: mydata.dtest.input[counter], target: mydata.dtest.target[counter]})
                    _accu_seed.append((1- error)*100)
                    # if newacc > oldaccu:
                    #     saver.save(sess, "models/model.ckpt")
                    #     oldaccu = newacc
                _accu.append(max(_accu_seed))
            print(device, keyword, max(_accu_seed))
            counter +=1
            output('Device: {0} , keyword: {1}, accuracy is :{2}'.format(device, keyword, np.average(_accu)), filename='res/newdeep_noend.txt', func='write')