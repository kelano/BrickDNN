import tensorflow as tf

import models.regression_v1 as dnn_module
import models.regression_v1_dropout as dnn_dropout_module
import models.ten_cents_a_piece as ten_cent_module
import models.regression_linear as dnn_linear_module

from dataset.dataset import *
from dataset.bow import *


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

theme_bow_vec = get_bow_vecs()

X_cols = [0, 1, 2]
# X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
Y_col = [3]
# Y_col = [len(theme_bow_vec) + 2]

input_size = len(X_cols)
inner_size = input_size * 4
output_size = 1

hm_epochs = 4000
eval_freq = 2000

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

norm_fn_map = {}
if normalize_on:
    norm_fn_map = normalize_data(np_dataset, X_cols)

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    print sess.run(tf.global_variables())

    for t_v in tf.trainable_variables():
        print t_v, sess.run(t_v)

    while True:
        # get data cols
        X = []
        x_data = []
        for X_col in X_cols:
            feat = feats[X_col]
            prompt = 'Enter ' + feat
            value = raw_input(prompt)
            converted_value = feature_converter_dict[feat](value)
            if normalize_on:
                converted_value = norm_fn_map[X_col](converted_value)
            x_data.append(converted_value)

        X.append(x_data)
        X = np.array(X)

        x = tf.placeholder('float', [None, input_size])
        keep_prob = tf.placeholder(tf.float32)
        prediction_fn = dnn_model.__call__(x, keep_prob)
        preds = sess.run(prediction_fn, feed_dict={x: X, keep_prob: 1.0})
        print preds
