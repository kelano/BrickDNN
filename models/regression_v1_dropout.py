import tensorflow as tf


class Model:
    def __init__(self, input_size, inner_size, output_size, name='dnn_dropout'):
        with tf.variable_scope(name):
            self.hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_size, inner_size])),
                              'biases': tf.Variable(tf.random_normal([inner_size]))}
            self.output_layer = {'weights': tf.Variable(tf.random_normal([inner_size, output_size])),
                            'biases': tf.Variable(tf.random_normal([output_size])), }

    def __call__(self, data, keep_prob, *args, **kwargs):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l1_dropout = tf.nn.dropout(l1, keep_prob=keep_prob)
        output = tf.matmul(l1_dropout, self.output_layer['weights']) + self.output_layer['biases']
        return output
