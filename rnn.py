""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from dataset.bow import *
word_bow_vec = []
with open('seuss.txt', 'r') as input:
    all = input.read()
    split = all.split()
    word_bow_vec = get_bow_vec_raw(split)
    encoded_all = []
    for word in split:
        encoded_all.append(get_bow_encoding(word_bow_vec, word))
    print('ok')


class Seuss2:
    def __init__(self, split):
        self.X = np.array(split[:-1])
        self.Y = np.array(split[1:])

        self.size = len(self.X)

        self.train_size = int(self.size * .8)
        self.X_train = self.X[:self.train_size]
        self.Y_train = self.Y[:self.train_size]

        self.test_size = self.size - self.train_size
        self.X_test = self.X[self.train_size:]
        self.Y_test = self.Y[self.train_size:]

        self.start = 0

    def next_batch(self, batch_size, timesteps):
        end = self.start + batch_size
        if end > self.train_size:
            remaining = end - self.train_size
            X_batch = self.X_train[self.start:self.train_size]
            X_batch = np.concatenate((X_batch, self.X_train[0:remaining]))
            Y_batch = self.Y_train[self.start:self.train_size]
            Y_batch = np.concatenate((Y_batch, self.Y_train[0:remaining]))
            # Y_batch = self.Y_train[remaining:remaining+1]
            self.start = remaining
        else:
            X_batch = self.X_train[self.start:end]
            Y_batch = self.Y_train[self.start:end]
            # Y_batch = self.Y_train[end:end+1]
            self.start = end
        return X_batch, Y_batch[timesteps-1::timesteps]

    def test(self, timesteps):
        return self.X_train, self.Y_train[::timesteps]


class Seuss:
    def __init__(self, split):
        self.X = np.array(split[:-1])
        self.Y = np.array(split[1:])

        self.size = len(self.X)

        self.train_size = int(self.size * .8)
        self.X_train = self.X[:self.train_size]
        self.Y_train = self.Y[:self.train_size]

        self.test_size = self.size - self.train_size
        self.X_test = self.X[self.train_size:]
        self.Y_test = self.Y[self.train_size:]

        self.start = 0

    def next_batch(self, batch_size, timesteps):
        end = self.start + batch_size
        if end > self.train_size:
            remaining = end - self.train_size
            X_batch = self.X_train[self.start:self.train_size]
            X_batch = np.concatenate((X_batch, self.X_train[0:remaining]))
            Y_batch = self.Y_train[self.start:self.train_size]
            Y_batch = np.concatenate((Y_batch, self.Y_train[0:remaining]))
            # Y_batch = self.Y_train[remaining:remaining+1]
            self.start = remaining
        else:
            X_batch = self.X_train[self.start:end]
            Y_batch = self.Y_train[self.start:end]
            # Y_batch = self.Y_train[end:end+1]
            self.start = end
        return X_batch, Y_batch[timesteps-1::timesteps]

    def test(self, timesteps):
        return self.X_train, self.Y_train[::timesteps]


dataset = Seuss(encoded_all)



# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 10#28
display_step = 1000

# Network Parameters
num_input = len(word_bow_vec)#28 # MNIST data input (img shape: 28*28)
timesteps = 5#2#28 # timesteps
num_hidden = num_input * 2#128 # hidden layer num of features
num_classes = len(word_bow_vec) # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, None, num_input])

print(X.shape)

x = tf.unstack(X, timesteps, 1)

print(x[0].shape, len(x))
# quit()

Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def genWord(bow_vec, p=None):
    pred_word_idx = np.random.choice(len(bow_vec), p=p)
    pred_word = bow_vec[pred_word_idx]
    pred_word_vec = np.array(get_bow_encoding(bow_vec, pred_word)).reshape(1, len(word_bow_vec))
    return pred_word, pred_word_vec

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x, batch_y = dataset.next_batch(batch_size)
        batch_x, batch_y = dataset.next_batch(batch_size, timesteps)

        for yi in range(0, batch_y.shape[0]):
            for xi in range(yi * timesteps, yi * timesteps + timesteps):
                xidx = np.argmax(batch_x[xi])
                print(word_bow_vec[xidx])
            yidx = np.argmax(batch_y[yi])
            print('=>' + word_bow_vec[yidx])



        # Reshape data to get 28 seq of 28 elements

        # print(batch_x.shape)
        batch_x = batch_x.reshape((-1, timesteps, num_input))
        # print(batch_x.shape)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            sentence = []
            sentence_len = 10

            word_vecs = []
            for i in range(0, timesteps):
                word, word_vec = genWord(word_bow_vec)
                word_vecs.append(word_vec)
                # sentence.append(word)

            for word_index in range(0, sentence_len):
                # print(word)
                # get next word

                concat = np.concatenate(word_vecs[-timesteps:], axis=0)

                word_vec_input = concat.reshape((1, timesteps, num_input))
                pred = sess.run(prediction, feed_dict={X: word_vec_input})

                # pred_word_idx = np.argmax(pred)
                preds = np.ravel(pred)
                word, word_vec = genWord(word_bow_vec, preds)
                word_vecs.append(word_vec)
                sentence.append(word)

            print(' '.join(sentence))

            # test_data = dataset.X_test.reshape((-1, timesteps, num_input))
            # test_label = dataset.Y_test
            # print("Testing Accuracy:", \
            #       sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

            print("Optimization Finished!")

            # Calculate accuracy for 128 mnist test images
            test_len = 128
            # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            # test_label = mnist.test.labels[:test_len]

            test_X, test_Y = dataset.test(timesteps)
            test_data = test_X.reshape((-1, timesteps, num_input))
            test_label = test_Y
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))