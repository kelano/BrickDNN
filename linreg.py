import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np

from dataset.dataset import *


# Get the data
total_features, total_prices = load_boston(True)


# Keep 300 samples for training
train_features = scale(total_features[:300])
train_prices = total_prices[:300]

# Keep 100 samples for validation
valid_features = scale(total_features[300:400])
valid_prices = total_prices[300:400]

# Keep remaining samples as test set
test_features = scale(total_features[400:])
test_prices = total_prices[400:]



feature_converter_dict = {
    "Minifigs": lambda str_value: 0 if str_value == '' else float(str_value),
    "Pieces": lambda str_value: None if str_value == '' else (float(str_value) if float(str_value) > 50 else None),
    "USPrice": lambda str_value: None if str_value == '' else (float(str_value) if float(str_value) > 0 else None),
    "CAPrice": lambda str_value: None if str_value == '' else float(str_value),
    # "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
    "Year": lambda str_value: None if str_value == '' else float(str_value)
}

feature_augment_dict = {
    "Minifigs": lambda str_value, scale: 0 if str_value == '' else float(str_value) * scale,
    "Pieces": lambda str_value, scale: None if str_value == '' else float(str_value) * scale,
    "USPrice": lambda str_value, scale: None if str_value == '' else float(str_value) * scale,
    "CAPrice": lambda str_value, scale: None if str_value == '' else float(str_value) * scale
}

#['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
# 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
feats = ["Pieces", "USPrice"]

# theme_bow_vec = get_bow_vecs()

# X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
X_cols = [0]#, 1, 2]
# Y_col = [len(theme_bow_vec) + 2]
Y_col = [1]
close_threshold = 10
piece_index = 0

feats_to_cols = {
    "Pieces": [0],
    # "Minifigs": [1],
    # "Year": [2],
    "USPrice": [1]
}

input_size = len(X_cols)

n_nodes_hl1 = input_size * 4
# n_nodes_hl2 = 10
# n_nodes_hl3 = 10

n_classes = 1
batch_size = 100

hm_epochs = 300

# train_neural_network(x)
np_dataset = get_datasets(feats, feature_converter_dict, X_cols, Y_col)#[range(0, 10), :]

# plt.scatter(np_dataset[:, 0], np_dataset[:, 1])
# plt.show()

# np_dataset = augment_dataset(np_dataset, feature_augment_dict, feats_to_cols)

normalize_on = True


if normalize_on:
    normalize_data(np_dataset, X_cols)

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)





train_features, train_prices, valid_features, valid_prices, test_features, test_prices =\
    split_dataset(np_dataset, X_cols, Y_col)






w = tf.Variable(tf.truncated_normal([input_size, 1], mean=0.0, stddev=1.0, dtype=tf.float64))
b = tf.Variable(tf.zeros(1, dtype=tf.float64))


def calc(x, y):
    # Returns predictions and error
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))

    return [ predictions, error ]


y, cost = calc(train_features, train_prices)
# Feel free to tweak these 2 values:
learning_rate = 0.01
epochs = 100000
points = [[], []] # You'll see later why I need this

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    # plt.axis([0, 500, -100, 100])
    # plt.ion()
    for i in list(range(epochs)):
        # print i

        sess.run(optimizer)

        if i % 10 == 0.:
            preds = sess.run(y)#np.transpose(sess.run(y))
            Y = train_prices
            abs_diff = np.abs(Y - preds)
            good_diff = np.where(abs_diff < 10.0)[0]

            accuracy = float(len(good_diff)) / float(len(abs_diff))
            # print 'epoch ' + str(i) + ' acc ' + str(accuracy)

            points[0].append(i+1)
            # points[1].append(sess.run(cost))
            points[1].append(accuracy)

        if i % 100 == 0:
            print(i, sess.run(cost))
            preds = sess.run(y)#np.transpose(sess.run(y))
            Y = train_prices
            diff = Y - preds
            abs_diff = np.abs(y - preds)
            good_diff = np.where(abs_diff < 10.0)[0]

            accuracy = float(len(good_diff)) / float(len(diff))


            # plt.clf()
            # plt.axis([0,500,-100,100])
            # plt.scatter(preds, Y - preds)
            # plt.pause(0.05)

    save_path = saver.save(sess, "model.ckpt")

    plt.plot(points[0], points[1], 'r--')
    plt.axis([0, epochs, 0, 1])
    plt.show()

    valid_cost = calc(valid_features, valid_prices)[1]

    print('Validation error =', sess.run(valid_cost), '\n')

    test_cost = calc(test_features, test_prices)[1]

    for var in tf.global_variables():
        print var, sess.run(var)


    # print('Test error =', sess.run(test_cost), '\n')

