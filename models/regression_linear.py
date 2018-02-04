import tensorflow as tf


class Model:
    def __init__(self, input_size, output_size, name='dnn_linear'):
        with tf.variable_scope(name):
            self.w = tf.Variable(tf.truncated_normal([input_size, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
            self.b = tf.Variable(tf.zeros(1, dtype=tf.float32))

    def __call__(self, data, *args, **kwargs):
        # l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        # l1 = tf.nn.relu(l1)
        # output = tf.matmul(data, self.output_layer['weights']) + self.output_layer['biases']
        # return output
        return tf.add(self.b, tf.matmul(data, self.w))
