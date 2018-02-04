import tensorflow as tf

import models.regression_v1 as dnn_module
import models.regression_v1_dropout as dnn_dropout_module
import models.ten_cents_a_piece as ten_cent_module
import models.regression_linear as dnn_linear_module

from matplotlib import pyplot as plt

from dataset.dataset import *
from dataset.bow import *


def evaluate(sess, model, X, Y, plot_res=False):
    x = tf.placeholder('float', [None, input_size])
    keep_prob = tf.placeholder(tf.float32)
    prediction_fn = model.__call__(x, keep_prob)
    if plot_res:
        print X[range(0, 10)]
    preds = sess.run(prediction_fn, feed_dict={x: X, keep_prob: 1.0})

    diff = np.abs(np.subtract(preds, Y))

    good_diff = np.where(diff < close_threshold)[0]
    print 'accuracy: ', float(len(good_diff)) / float(len(diff))

    if plot_res:
        print Y[range(0, 10)]
        print preds[range(0, 10)]
        print np.subtract(preds, Y)[range(0, 10)]
        # plt.scatter(preds, np.subtract(preds, Y))
        # plt.show()

    return float(len(good_diff)) / float(len(diff))


def train_neural_network(eval_map):
    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)

    cost_map = {}
    optimizer_map = {}

    for key, value in eval_map.iteritems():
        prediction = value.__call__(x, keep_prob)
        cost_map[key] = tf.reduce_mean(tf.square(prediction - y))
        # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
        if "dnn" in key:
            optimizer_map[key] = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_map[key])
            # optimizer_map[key] = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_map[key])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        learning_results = {}
        eval_results = {}

        for epoch in range(hm_epochs):
            epoch_x, epoch_y = train_X, train_Y
            for key, value in eval_map.iteritems():
                cost = cost_map[key]
                if "dnn" in key:
                    optimizer = optimizer_map[key]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})
                    epoch_loss = c
                else:
                    epoch_loss = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})

                if (epoch % 100) == 0:
                    print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
                # key += '_cost'
                if key not in learning_results:
                    learning_results[key] = []
                learning_results[key].append(epoch_loss)
                if epoch == (hm_epochs - 1):
                    saver.save(sess, "/tmp/model.ckpt")

            if (epoch % eval_freq) == 0:
                for key, value in eval_map.iteritems():
                    print('Evaluating ', key)
                    # acc = evaluate(sess, value, dev_X, dev_Y)
                    eval_loss = sess.run(cost_map[key], feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
                    if key not in eval_results:
                        eval_results[key] = []
                    eval_results[key].append(eval_loss)

        # save models
        for key, value in eval_map.iteritems():
            if 'dnn' in key:
                saver.save(sess, "model.ckpt")

        learning_x_axis = np.arange(0, hm_epochs, 1)
        plt.figure(1)
        plt.title('Learning Curves')
        for key, val in learning_results.iteritems():
            print len(learning_x_axis), len(val)
            plt.plot(learning_x_axis, val, shapes[eval_map.keys().index(key)] + '--', label=key + '_train')
        plt.legend()

        epoch_x_axis = np.arange(0, hm_epochs, eval_freq)
        # plt.figure(2)
        # plt.title('Accuracy (within $10)')
        for key, val in eval_results.iteritems():
            print len(epoch_x_axis), len(val)
            plt.plot(epoch_x_axis, val,  shapes[eval_map.keys().index(key)] + '-', label=key + '_evaluation')
        plt.legend()
        plt.show()


feature_converter_dict = {
    "Minifigs": lambda str_value: 0 if str_value == '' else float(str_value),
    "Pieces": lambda str_value: None if str_value == '' else float(str_value),
    "USPrice": lambda str_value: None if str_value == '' else float(str_value),
    "CAPrice": lambda str_value: None if str_value == '' else float(str_value),
    "UKPrice": lambda str_value: None if str_value == '' else float(str_value),
    "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
    "Year": lambda str_value: None if str_value == '' else float(str_value)
}

feature_augment_dict = {
    "Minifigs": lambda str_value, scale: 0 if str_value == '' else float(str_value) * scale,
    "Pieces": lambda str_value, scale: None if str_value == '' else float(str_value) * scale,
    "USPrice": lambda str_value, scale: None if str_value == '' else float(str_value) * scale,
    "CAPrice": lambda str_value, scale: None if str_value == '' else float(str_value) * scale,
    "UKPrice": lambda str_value, scale: None if str_value == '' else float(str_value) * scale
}

# ['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
# 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
feats = ["Pieces", "Minifigs", "Year", "USPrice"]

feats_to_cols = {
    "Pieces": [0],
    "Minifigs": [1],
    "Year": [2],
    "USPrice": [3]
}

theme_bow_vec = get_bow_vecs('Theme')

X_cols = [0, 1, 2]
# X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
Y_col = [3]
# Y_col = [len(theme_bow_vec) + 2]

input_size = len(X_cols)
inner_size = input_size * 4
output_size = 1

hm_epochs = 10000
eval_freq = 100

close_threshold = 10

np_dataset = get_datasets(feats, feature_converter_dict, X_cols, Y_col)

normalize_on = True

ten_cent_model = ten_cent_module.Model(np_dataset, piece_index=feats.index('Pieces'), normalize_on=normalize_on)
dnn_model = dnn_module.Model(input_size, inner_size, output_size)
dnn_linear_model = dnn_linear_module.Model(input_size, output_size)
dnn_dropout_model = dnn_dropout_module.Model(input_size, inner_size, output_size)

eval_map = {
    "ten_cent": ten_cent_model,
    "dnn": dnn_model,
    "dnn_dropout": dnn_dropout_model,
    "dnn_linear": dnn_linear_model
}

if normalize_on:
    normalize_data(np_dataset, X_cols)

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)

shapes = ('r', 'g', 'b', 'y')

train_neural_network(eval_map)
# import analysis
# analysis.perform_analysis(X_cols, Y_col, np_dataset)