import csv
import numpy as np


def get_datasets(feats, feature_converter_dict, X_cols, Y_col, debug=False):
    with open('data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        first = True
        data = []
        for row in reader:
            if first:
                # ignore, header
                first = False
                headers = row
            else:
                data.append(row)
        # print headers
        # print data[0]

        filtered_data = []
        for data_row in data:
            filtered_data_row = []
            for feat in feats:
                orig_index = headers.index(feat)
                orig_value = data_row[orig_index]
                new_value = feature_converter_dict[feat](orig_value)
                if new_value is None:
                    if debug:
                        print 'filtered from', feat, ' ' + orig_value + ' : ', data_row
                    continue
                filtered_data_row.extend(new_value) if type(new_value) is list else filtered_data_row.append(new_value)
            if len(filtered_data_row) == len(X_cols) + len(Y_col):
                filtered_data.append(filtered_data_row)
        np_data = np.array(filtered_data)

        # print np_data
        print 'Data set size Filtered: {} Unfiltered: {}'.format(len(data), len(np_data))
        return np_data


def split_dataset(dataset, X_cols, Y_col):
    np.random.shuffle(dataset)

    total_size = len(dataset)

    train = int(total_size * .8)
    val = int((total_size - train)/2.0)
    test = total_size - train - val
    print total_size, train, val, test

    train_data = dataset[range(0, train)]
    dev_data = dataset[range(train, train + val)]
    test_data = dataset[range(train + val, len(dataset))]

    return train_data[:, X_cols], train_data[:, Y_col],\
           dev_data[:, X_cols], dev_data[:, Y_col],\
           test_data[:, X_cols], test_data[:, Y_col]


def normalize_data(dataset, X_cols):
    norm_fn_map = {}
    for X_col in X_cols:
        x_data = dataset[:, X_col]
        x_min = x_data.min()
        x_max = x_data.max()
        print X_col, x_min, x_max
        norm_fn_map[X_col] = lambda x, x_min=x_min, x_max=x_max: np.divide(np.subtract(x, x_min), x_max - x_min)
        normed = norm_fn_map[X_col](x_data)
        # normed = np.divide(np.subtract(x_data, x_min), x_max - x_min)
        dataset[:, X_col] = normed
    return norm_fn_map


def augment_dataset(dataset, feature_augment_dict, feats_to_cols):
    orig_len = len(dataset)
    all_augmented_data = []
    for i in range(0, orig_len):
        orig_data = dataset[i]
        augmented_data = np.copy(orig_data)
        for feat, cols in feats_to_cols.iteritems():
            if feat in feature_augment_dict:
                for col in cols:
                    augmented_data[col] = feature_augment_dict[feat](augmented_data[col], 2.0)
        all_augmented_data.append(augmented_data)

    all_augmented_data = np.array(all_augmented_data)
    dataset = np.append(dataset, all_augmented_data, axis=0)
    return dataset


