import functools
import re
# import sets
import tensorflow as tf
from tensorflow.nn import rnn_cell, dynamic_rnn

from utils.datas import Data

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

    def __init__(self, data, target, num_hidden=500, num_layers=2):
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
        cell = rnn_cell.LSTMCell(self._num_hidden)
        # cell = rnn_cell.MultiRNNCell([cell for _ in range(self._num_layers)])
        # tf.contrib.rnn.AttentionCellWrapper(cell, 20)
        # cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.7)
        # init_state = cell.zero_state(batchsize, tf.float32)
        output, _ = dynamic_rnn(
            cell,
            self.data,
            dtype=tf.float32,
            # initial_state=init_state,
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
        # w = tf.constant(mydata.weights, dtype= tf.float32)
        # prediction =  self.prediction * w
        cross_entropy = self.target * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0))
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
        weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight, name='weight'), tf.Variable(bias, name='bias')


if __name__ == '__main__':
    import numpy as np
    devicelist = ['Mac Laptop']
    keywords = ['']
    batchsize = 10
    counter = -1
    seeds = [0, 12, 21, 32, 45, 64, 77, 98, 55, 120]
    for seed in seeds:
        learning_rate = 0.01
        mydata = Data(devicelist, keywords, seed , deep = True, notool=False)
        tf.reset_default_graph()
        trainlen, length, tool_number = np.shape(mydata.dtrain.input[counter])
        data = tf.placeholder(tf.float32, [None, length, tool_number], name='input')
        target = tf.placeholder(tf.float32, [None, length, tool_number], name='target')
        model = VariableSequenceLabelling(data, target)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        old_accu = 0
        best_epoch = 0
        print('start training with lr = {0}'.format(learning_rate))
        early = 20
        for epoch in range(300):
            for n in range(int(trainlen / batchsize)):
                sess.run(model.optimize, {data: mydata.dtrain.input[counter][n * batchsize: (n + 1) * batchsize],
                                          target: mydata.dtrain.target[counter][n * batchsize: (n + 1) * batchsize]})
            error = sess.run(model.error, {data: mydata.dtest.input[counter], target: mydata.dtest.target[counter]})
            new_accu = (1 - error) * 100
            print('Epoch {:2d} accuracy {:3.1f}%'.format(epoch + 1, new_accu))
            if new_accu > old_accu and epoch > 5:
                saver.save(sess, "models/models_lstm/withoutnotool/Maclaptob_{0}.ckpt".format(seed), write_meta_graph=False)
                old_accu = new_accu
                best_epoch = epoch
            if epoch - best_epoch > early : break #early stopping
            if (epoch+1)% 30==0:
                learning_rate /= 2
                print('learning rate reduced to {0}'.format(learning_rate))
            if new_accu == 0 and epoch > 5:
                best_epoch = epoch
                sess.run(tf.global_variables_initializer())
                print('zero accuracy')
                learning_rate /= 2
                early = 70
                print('learning rate reduced to {0}'.format(learning_rate))