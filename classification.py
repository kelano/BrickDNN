import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import models.regression_v1_dropout as dnn_module
import models.regression_v1 as dnn_module_nodrop

from matplotlib import pyplot as plt

from dataset.dataset import *
from dataset.bow import *


def train_epoch(key, train_X, train_Y, batch_size, current_learning_rate, batch_losses, optimizer, cost, sess, x, y, keep_prob, learning_rate):
    # x = tf.placeholder('float', [None, input_size])
    # y = tf.placeholder('float')
    # keep_prob = tf.placeholder(tf.float32)
    # learning_rate = tf.placeholder(tf.float32, shape=[])

    # train batch
    batch_count = 0
    batches = get_batch_indices(batch_size, len(train_X))
    for batch in batches:
        batch_x = train_X[batch[0]:batch[1]]
        batch_y = train_Y[batch[0]:batch[1]]

        if "dnn" in key:
            _, batch_loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5,
                                                          learning_rate: current_learning_rate})
        else:
            batch_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        # print('batch loss (%d/%d): %f' % (batch_count, len(batches), batch_loss))
        batch_count = batch_count + 1
        batch_losses.append(batch_loss)

    return len(batches)


def run_xval(key, dev_X, dev_Y, xval_losses, prediction, cost, sess, x, y, keep_prob):

    # print(('Evaluating ', key))
    # acc = evaluate(sess, value, dev_X, dev_Y)
    preds = sess.run(prediction, feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
    xval_loss = sess.run(cost, feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
    corr = len(np.where(np.equal(np.argmax(preds, axis=1), np.argmax(dev_Y, axis=1)))[0])
    # print('Evaluating ', key, ':', corr, ' correct out of ', len(dev_Y))
    acc = float(corr) / len(dev_Y)
    print('XVAL Acc: %f (%d/%d) Loss: %f' % (acc, corr, len(dev_Y), xval_loss))
    xval_losses.append(xval_loss)
    return xval_loss, acc


def run_batch_eval(key, X, Y, batch_size, prediction, cost, sess, x, y, keep_prob):
    batch_losses = []
    tot_cor = 0
    tot = 0
    batches = get_batch_indices(batch_size, len(X))

    for batch in batches:
        batch_x = X[batch[0]:batch[1]]
        batch_y = Y[batch[0]:batch[1]]
        preds = sess.run(prediction, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        xval_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        corr = len(np.where(np.equal(np.argmax(preds, axis=1), np.argmax(batch_y, axis=1)))[0])
        tot_cor += corr
        tot += len(batch_y)
        batch_losses.append(xval_loss)

    mean_loss = float(np.mean(batch_losses))
    acc = float(tot_cor) / tot

    print('XVAL Acc: %f (%d/%d) Mean Loss: %f' % (acc, tot_cor, tot, mean_loss))
    return mean_loss, acc, batch_losses


def train_neural_network_new_map(eval_map):
    plt.figure(1)
    plt.title('Learning Curves')
    for key, model in eval_map.items():
        num_batches, batch_losses, xval_losses = train_neural_network_new(key, model)

        plt.plot([(float(x) / num_batches) for x in range(0, len(batch_losses))], batch_losses, linestyle='--',
                 color=shapes[list(eval_map).index(key)], label=key + '_train')

        plt.plot(range(0, len(xval_losses)), xval_losses, linestyle='-',
                 color=shapes[list(eval_map).index(key)],label=key + '_xval')

    plt.legend()

    # epoch_x_axis = np.arange(0, hm_epochs, eval_freq)
    # plt.figure(2)
    # plt.title('Accuracy')
    # for key, val in eval_results.items():
    #     print(len(epoch_x_axis), len(val))
    #     plt.plot(epoch_x_axis, val,  shapes[list(eval_map).index(key)] + '-', label=key + '_evaluation')
    # plt.legend()
    plt.show()


def train_neural_network_new(key, model):
    batch_size = 128

    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    current_learning_rate = initial_learning_rate
    tot_decays = 0

    prediction = model.__call__(x, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    if "dnn" in key:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # with tf.device('/device:CPU:0'):
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.restore(sess, "C:\code\BrickDNN\model.ckpt")

        epoch = 0
        stop_training = False
        xval_losses = []
        batch_losses = []

        while epoch < hm_epochs and not stop_training:
            print('Epoch', epoch)

            num_batches = train_epoch(key, train_X, train_Y, batch_size, current_learning_rate, batch_losses, optimizer,
                                      cost, sess, x, y, keep_prob, learning_rate)

            xval_loss, xval_acc, xval_losses = run_batch_eval(key, dev_X, dev_Y, batch_size, prediction, cost, sess, x, y, keep_prob)

            if len(xval_losses) > 1 and xval_losses[-1] > xval_losses[-2]:
                new_learning_rate = current_learning_rate * learning_rate_decay
                print('DECAYING RATE %f -> %f' % (current_learning_rate, new_learning_rate))
                tot_decays = tot_decays + 1
                if tot_decays > max_decays:
                    stop_training = True
                current_learning_rate = new_learning_rate

            epoch = epoch + 1

        # save models
        if 'dnn' in key:
            saver.save(sess, "C:\code\BrickDNN\model.ckpt")

    return num_batches, batch_losses, xval_losses


def train_neural_network(eval_map):

    batch_size = 128

    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    current_learning_rate = initial_learning_rate
    tot_decays = 0

    cost_map = {}
    optimizer_map = {}
    prediction = {}

    for key, value in eval_map.items():
        prediction[key] = value.__call__(x, keep_prob)
        cost_map[key] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction[key], labels=y))
        if "dnn" in key:
            optimizer_map[key] = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_map[key])
            # optimizer_map[key] = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_map[key])

    # names = [n for n in tf.get_default_graph().as_graph_def().node]
    # name2s = [n for n in tf.get_default_graph().as_graph_def()]
    # print(names)
    # quit()

    # with tf.device('/device:CPU:0'):
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # saver.restore(sess, "C:\code\BrickDNN\model.ckpt")

        learning_results = {}
        eval_results = {}
        eval_loss = {}
        epoch = 0
        stop_training = False

        for key, value in eval_map.items():
            while epoch < hm_epochs and not stop_training:
                print('Epoch', epoch)
                epoch_x, epoch_y = train_X, train_Y
                    # print('Training ', key)

                # train batch
                batch_count = 0
                batches = get_batch_indices(batch_size, len(epoch_x))
                for batch in batches:
                    batch_x = epoch_x[batch[0]:batch[1]]
                    batch_y = epoch_y[batch[0]:batch[1]]

                    cost = cost_map[key]
                    if "dnn" in key:
                        optimizer = optimizer_map[key]
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5,
                                                                      learning_rate: current_learning_rate})
                        # print(_)
                        # print(c)
                        # quit()
                        batch_loss = c
                    else:
                        batch_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                    print('batch loss (%d/%d): %f' % (batch_count, len(batches), batch_loss))
                    batch_count = batch_count + 1
                    if key not in learning_results:
                        learning_results[key] = []
                    learning_results[key].append(batch_loss)



                # if epoch == (hm_epochs - 1):
                #     saver.save(sess, "/tmp/model.ckpt")

                if (epoch % eval_freq) == 0:
                    for key, value in eval_map.items():
                        print(('Evaluating ', key))
                        # acc = evaluate(sess, value, dev_X, dev_Y)
                        preds = sess.run(prediction[key], feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})
                        xval_loss = sess.run(cost_map[key], feed_dict={x: dev_X, y: dev_Y, keep_prob: 1.0})



                        corr = len(np.where(np.equal(np.argmax(preds, axis=1), np.argmax(dev_Y, axis=1)))[0])
                        print(corr, ' correct out of ', len(dev_Y))
                        acc = float(corr) / len(dev_Y)
                        if key not in eval_results:
                            eval_results[key] = []



                        eval_results[key].append(acc)
                        if key not in eval_loss:
                            eval_loss[key] = []


                        # adaptive LR
                        if len(eval_loss[key]) > 0 and xval_loss > eval_loss[key][-1]:
                            new_learning_rate = current_learning_rate * learning_rate_decay
                            print('DECAYING RATE %f -> %f' % (current_learning_rate, new_learning_rate))
                            tot_decays = tot_decays + 1
                            if tot_decays > max_decays:
                                stop_training = True
                            current_learning_rate = new_learning_rate


                        eval_loss[key].append(xval_loss)

                epoch = epoch + 1

        # save models
        for key, value in eval_map.items():
            if 'dnn' in key:
                saver.save(sess, "C:\code\BrickDNN\model.ckpt")

        learning_x_axis = np.arange(0, hm_epochs, 1)
        plt.figure(1)
        plt.title('Learning Curves')
        for key, val in learning_results.items():
            print(len(learning_x_axis), len(val))
            xval = eval_loss[key]
            plt.plot([float(x) / len(batches) for x in range(0, len(val))], val, shapes[list(eval_map).index(key)] + '--', color='r', label=key + '_train')
            plt.plot(range(0, len(eval_loss[key])), xval, shapes[list(eval_map).index(key)] + '-', color='g', label=key + '_xval')
        plt.legend()

        # epoch_x_axis = np.arange(0, hm_epochs, eval_freq)
        # plt.figure(2)
        # plt.title('Accuracy')
        # for key, val in eval_results.items():
        #     print(len(epoch_x_axis), len(val))
        #     plt.plot(epoch_x_axis, val,  shapes[list(eval_map).index(key)] + '-', label=key + '_evaluation')
        # plt.legend()
        plt.show()


def test_neural_network(eval_map):
    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)
    prediction = {}

    for key, value in eval_map.items():
        prediction = value.__call__(x, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

        # with tf.device('/device:CPU:0'):
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            saver.restore(sess, "C:\code\BrickDNN\model.ckpt")
            # saver.restore(sess, "C:\code\BrickDNN\model.ckpt")
            print(sess.run(tf.global_variables()))

            # checkpoint_path = os.path.join("C:\code\BrickDNN\model.ckpt")
            print_tensors_in_checkpoint_file("C:\code\BrickDNN\model.ckpt", all_tensors=True, all_tensor_names=True,
                                             tensor_name='')

            batch_size = 128
            test_loss, test_acc, test_losses = run_batch_eval(key, test_X, test_Y, batch_size, prediction, cost, sess, x,
                                                 y, keep_prob)


def live_test_network(model):
    feature_converter_dict = {
        # "Theme": lambda str_value: list((int(str_value != 'Star Wars'), int(str_value == 'Star Wars'))),
        "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
        # "Name": lambda str_value: get_bow_encoding(set_name_bow_vec, str_value, split=True, stem=True),
        "Name": lambda str_value: fasttext.get_fasttext_encoding(fasttext_data, str_value)
    }

    feature_augment_dict = {
    }

    # ['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
    # 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
    feats = ["Name", "Theme"]

    import fasttext
    print('loading fasttest')
    fasttext_data = fasttext.load_vectors()
    print('done loading')

    # set_name_bow_vec_2 = get_bow_vecs('Name', split=True, stem=False)
    # print len(set_name_bow_vec)
    # print len(set_name_bow_vec_2)

    feats_to_cols = {
        # "Name": list(range(0, len(set_name_bow_vec)))
        "Name": list(range(0, 300))
        # "Theme": range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec))
    }

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "C:\code\BrickDNN\model.ckpt")
        print(sess.run(tf.global_variables()))

        for t_v in tf.trainable_variables():
            print(t_v, sess.run(t_v))

        while True:
            # get data cols
            X = np.zeros(len(X_cols))
            x_data = []
            for feat, cols in feats_to_cols.items():
                prompt = 'Enter ' + feat
                value = input(prompt)
                converted_value = feature_converter_dict[feat](value)
                # if normalize_on:
                #     converted_value = norm_fn_map[X_col](converted_value)
                X[cols] = converted_value

            # X = np.transpose(X)
            # X = np.array(X)

            # print(np.where(X > 0))
            # print(set_name_bow_vec[np.where(X > 0)])
            # for x in np.where(X > 0)[0]:
            #     print(set_name_bow_vec[x])
            X.shape = (1, len(X_cols))
            x = tf.placeholder('float', [None, input_size])
            keep_prob = tf.placeholder(tf.float32)
            prediction_fn = dnn_model.__call__(x, keep_prob)
            sigmoid_fn = tf.nn.softmax(prediction_fn)
            presoft, preds = sess.run([prediction_fn, sigmoid_fn], feed_dict={x: X, keep_prob: 1.0})
            # print(presoft)
            # print(preds)
            # print(np.argmax(preds, axis=1)[0])
            print(theme_bow_vec[np.argmax(preds, axis=1)[0]])
            print(np.argsort(preds)[:, list(range(0, 5))])
            for id in np.fliplr(np.argsort(preds))[:, list(range(0, 5))].tolist()[0]:
                print(id, theme_bow_vec[id], preds[0, id])


def build_dataset():

    feature_augment_dict = {
    }

    # ['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
    # 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
    feats = ["Name", "Theme"]


    import fasttext
    print('loading fasttest')
    word_2_vec, word_to_idx, idx_to_vec, tf_embedding = fasttext.load_vectors()
    print('done loading')

    # set_name_bow_vec_2 = get_bow_vecs('Name', split=True, stem=False)
    # print len(set_name_bow_vec)
    # print len(set_name_bow_vec_2)


    feature_converter_dict = {
        # "Theme": lambda str_value: list((int(str_value != 'Star Wars'), int(str_value == 'Star Wars'))),
        "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
        # "Name": lambda str_value: get_bow_encoding(set_name_bow_vec, str_value, split=True, stem=True),
        "Name": lambda str_value: fasttext.get_fasttext_encoding(word_2_vec, str_value)
    }

    feats_to_cols = {
        # "Name": list(range(0, len(set_name_bow_vec)))
        "Name": list(range(0, 300))
        # "Theme": range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec))
    }

    np_dataset = get_datasets(feats, feature_converter_dict, X_cols, Y_col)
    # np_dataset = np_dataset[0:100, :]

    # tot = np.zeros(len(theme_bow_vec))
    # for np_data in np_dataset:
    #     tot += np_data[feats_to_cols['Theme']]

    train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)

    save_dataset(train_X, train_Y, dev_X, dev_Y, test_X, test_Y)
    # quit()


hm_epochs = 1000
eval_freq = 1
initial_learning_rate = 0.01
learning_rate_decay = 0.1
max_decays = 3

theme_bow_vec = get_bow_vecs('Theme')
set_name_bow_vec = get_bow_vecs('Name', split=True, stem=True)

# X_cols = list(range(0, len(set_name_bow_vec)))
X_cols = list(range(0, 300))
# X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
# Y_col = list(range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec)))
Y_col = list(range(300, 300 + len(theme_bow_vec)))
# Y_col = list(range(len(set_name_bow_vec), len(set_name_bow_vec) + 2))
# Y_col = [len(theme_bow_vec) + 2]

input_size = len(X_cols)
inner_size = input_size * 2
output_size = len(Y_col)

dnn_model_dropout = dnn_module.Model(input_size, inner_size, output_size)
dnn_model = dnn_module_nodrop.Model(input_size, inner_size, output_size)

eval_map = {
    "dnn_dropout": dnn_model_dropout,
    # "dnn": dnn_model,
}

build_dataset()
train_X, train_Y, dev_X, dev_Y, test_X, test_Y = load_dataset()


# test_dist_dict = {}
# for test_y in test_Y:
#     theme = theme_bow_vec[np.argmax(test_y)]
#     if theme not in test_dist_dict:
#         test_dist_dict[theme] = 0
#     test_dist_dict[theme] += 1

shapes = ('r', 'g', 'b', 'y')

# train_neural_network(eval_map)
# train_neural_network_new_map(eval_map)
test_neural_network(eval_map)
# live_test_network(dnn_model)


# import analysis
# analysis.perform_analysis(X_cols, Y_col, np_dataset, set_name_bow_vec)
#
print('done')