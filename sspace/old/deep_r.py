import functools
import re
from utils.util import output
import tensorflow as tf
from tensorflow.nn import rnn_cell, dynamic_rnn
from utils.datas import Data
from utils.util import mean_confidence_interval


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

    def __init__(self, data, target, toolmask = None, num_hidden=500, num_layers=4):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.toolmask = toolmask
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
        cell = rnn_cell.LSTMCell(self._num_hidden)
        # cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.5)
        output, _ = dynamic_rnn(
            cell,
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
        preds = self.prediction
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(preds, 2))
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
    filename = 'res/deep/withoutnotool/nlstm_'
    batchsize = 10
    accu_list = []
    verbose = None
    seeds = [0, 12, 21, 32, 45, 64, 77, 98] # 55, 120, 10, 20, 30, 40, 60, 70, 80]
    for i, seed in enumerate(seeds):
        mydata = Data(seed, deep = True)
        tf.reset_default_graph()
        _ , length, tool_number = np.shape(mydata.dtrain.input)
        data = tf.placeholder(tf.float32, [None, length, tool_number], name='input')
        target = tf.placeholder(tf.float32, [None, length, tool_number], name='target')
        # toolmask = tf.placeholder(tf.float32, [tool_number], name='toolmask')

        model = VariableSequenceLabelling(data, target)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(sess, "/hri/localdisk/nnabizad/models/deep_r/Maclaptob_{0}.ckpt".format(seed))
        # counter = 0
        error = sess.run(model.error, {data: mydata.dtest.input, target: mydata.dtest.target})
        acc = ((1 - error) * 100)
        accu_list.append(acc)
        print(acc)