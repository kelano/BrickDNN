import tensorflow as tf


class Model:
    def __init__(self, np_dataset, piece_index, normalize_on):
        data = np_dataset[:, piece_index]
        self.data_min = data.min()
        self.data_max = data.max()
        self.piece_index = piece_index
        self.normalize_on = normalize_on

    def __call__(self, data, *args, **kwargs):
        data_col = tf.reshape(tf.gather(tf.transpose(data), self.piece_index), [tf.shape(data)[0], 1])
        price_col = tf.scalar_mul(0.10, tf.ones([tf.shape(data)[0], 1]))
        if self.normalize_on:
            min_col = tf.scalar_mul(self.data_min, tf.ones([tf.shape(data)[0], 1]))
            diff_col = tf.scalar_mul(self.data_max - self.data_min, tf.ones([tf.shape(data)[0], 1]))
            return tf.multiply(tf.add(tf.multiply(data_col, diff_col), min_col), price_col)
        else:
            return tf.multiply(data_col, price_col)

