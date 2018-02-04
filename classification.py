import tensorflow as tf

import models.regression_v1 as dnn_module

# from matplotlib import pyplot as plt

from dataset.dataset import *
from dataset.bow import *


def train_neural_network(eval_map):
    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)

    cost_map = {}
    optimizer_map = {}
    prediction = {}

    for key, value in eval_map.iteritems():
        prediction[key] = value.__call__(x, keep_prob)
        cost_map[key] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction[key], labels=y))
        if "dnn" in key:
            optimizer_map[key] = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_map[key])
            # optimizer_map[key] = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_map[key])

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        saver.restore(sess, "model.ckpt")

        learning_results = {}
        eval_results = {}

        for epoch in range(hm_epochs):
            epoch_x, epoch_y = train_X, train_Y
            for key, value in eval_map.iteritems():
                cost = cost_map[key]
                print 'Training ', key
                if "dnn" in key:
                    optimizer = optimizer_map[key]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})
                    epoch_loss = c
                else:
                    epoch_loss = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})

                if (epoch % eval_freq) == 0:
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
                    preds = sess.run(prediction[key], feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
                    # eval_loss = sess.run(cost_map[key], feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
                    corr = len(np.where(np.equal(np.argmax(preds, axis=1), np.argmax(dev_Y, axis=1)))[0])
                    print corr, ' correct out of ', len(dev_Y)
                    acc = float(corr) / len(dev_Y)
                    if key not in eval_results:
                        eval_results[key] = []
                    eval_results[key].append(acc)

        # save models
        for key, value in eval_map.iteritems():
            if 'dnn' in key:
                saver.save(sess, "model.ckpt")

        # learning_x_axis = np.arange(0, hm_epochs, 1)
        # plt.figure(1)
        # plt.title('Learning Curves')
        # for key, val in learning_results.iteritems():
        #     print len(learning_x_axis), len(val)
        #     plt.plot(learning_x_axis, val, shapes[eval_map.keys().index(key)] + '--', label=key + '_train')
        # plt.legend()
        #
        # epoch_x_axis = np.arange(0, hm_epochs, eval_freq)
        # plt.figure(2)
        # plt.title('Accuracy')
        # for key, val in eval_results.iteritems():
        #     print len(epoch_x_axis), len(val)
        #     plt.plot(epoch_x_axis, val,  shapes[eval_map.keys().index(key)] + '-', label=key + '_evaluation')
        # plt.legend()
        # plt.show()

def live_test_network(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        print sess.run(tf.global_variables())

        for t_v in tf.trainable_variables():
            print t_v, sess.run(t_v)

        while True:
            # get data cols
            X = np.zeros(len(X_cols))
            x_data = []
            for feat, cols in feats_to_cols.iteritems():
                prompt = 'Enter ' + feat
                value = raw_input(prompt)
                converted_value = feature_converter_dict[feat](value)
                # if normalize_on:
                #     converted_value = norm_fn_map[X_col](converted_value)
                X[cols] = converted_value

            # X = np.transpose(X)
            # X = np.array(X)

            print np.where(X > 0)
            X.shape = (1, len(X_cols))
            x = tf.placeholder('float', [None, input_size])
            keep_prob = tf.placeholder(tf.float32)
            prediction_fn = dnn_model.__call__(x, keep_prob)
            sigmoid_fn = tf.nn.sigmoid(prediction_fn)
            preds = sess.run(prediction_fn, feed_dict={x: X, keep_prob: 1.0})
            print preds
            print np.argmax(preds, axis=1)[0]
            print theme_bow_vec[np.argmax(preds, axis=1)[0]]
            print np.argsort(preds)[:, range(0, 5)]
            for id in np.fliplr(np.argsort(preds))[:, range(0, 5)].tolist()[0]:
                print id, theme_bow_vec[id], preds[0, id]


feature_converter_dict = {
    "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
    "Name": lambda str_value: get_bow_encoding(set_name_bow_vec, str_value, split=True, stem=True)
}

feature_augment_dict = {
}

# ['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
# 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
feats = ["Name", "Theme"]

theme_bow_vec = get_bow_vecs('Theme')
set_name_bow_vec = get_bow_vecs('Name', split=True, stem=True)
# set_name_bow_vec_2 = get_bow_vecs('Name', split=True, stem=False)
# print len(set_name_bow_vec)
# print len(set_name_bow_vec_2)

feats_to_cols = {
    "Name": range(0, len(set_name_bow_vec))
    # "Theme": range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec))
}


X_cols = range(0, len(set_name_bow_vec))
# X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
Y_col = range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec))
# Y_col = [len(theme_bow_vec) + 2]

input_size = len(X_cols)
inner_size = input_size * 2
output_size = len(Y_col)

hm_epochs = 50
eval_freq = 1


np_dataset = get_datasets(feats, feature_converter_dict, X_cols, Y_col)
# np_dataset = np_dataset[0:100, :]


# tot = np.zeros(len(theme_bow_vec))
# for np_data in np_dataset:
#     tot += np_data[feats_to_cols['Theme']]


dnn_model = dnn_module.Model(input_size, inner_size, output_size)

eval_map = {
    "dnn": dnn_model,
}

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)

shapes = ('r', 'g', 'b', 'y')

train_neural_network(eval_map)
# import analysis
# analysis.perform_analysis(X_cols, Y_col, np_dataset)
# live_test_network(dnn_model)

print 'done'