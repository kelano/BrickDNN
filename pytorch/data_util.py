import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data as utils
import pytorch.elmo_scrap as elmo_util


def get_bow_vec():
    return []


def load_data():
    df = pd.read_csv('../data_201903.csv')

    print(df.columns)

    themes = df.Theme.unique()
    print(len(themes))
    # print(themes)

    themes = np.array(themes)

    # print(np.where(themes == 'Scala')[0][0])

    return df


def sentence_to_idx_rep(x, word_2_idx, pad_to=None):
    # print(x)
    sentence = str(x).lower()
    # print(sentence, len(sentence.split()))
    idxs = np.zeros(len(sentence.split())) if pad_to is None else np.zeros(pad_to)
    idx = 0
    for word in sentence.split():
        # print(word)
        if word in word_2_idx:
            idxs[idx] = word_2_idx[word]
        else:
            idxs[idx] = word_2_idx['UNK']
        idx += 1
    return idxs


def to_idx_rep(X, word_2_idx):
    sentence_lens = X.apply(lambda x: len(x.split()))
    max_len = sentence_lens.max()
    print(max_len)
    # exit()
    return X.apply(lambda x: sentence_to_idx_rep(x, word_2_idx, pad_to=max_len))


def sentence_to_vec_rep(x, word_2_vec, embedding_size=300, pad_to=None):
    # global ALL_INV, ALL_OOV

    sentence = str(x).lower()

    if pad_to is None:
        pad_to = len(sentence.split())
    sentence_emb = np.ones((pad_to, embedding_size), dtype=np.int64) * word_2_vec['<PAD>']
    idx = 0
    if len(sentence.split()) == 0:
        print('what', sentence)
        exit()
    # sentence = preprocess_sentence(sentence)
    for word in sentence.split():
        # if word == 'alexa':
        #     continue
        if word in word_2_vec:
            sentence_emb[idx] = word_2_vec[word]
        else:
            sentence_emb[idx] = word_2_vec['UNK']
        idx += 1
    return sentence_emb


def to_vec_rep(X, word_2_vec):
    sentence_lens = X.apply(lambda x: len(x.split()))
    max_len = sentence_lens.max()
    print(max_len)
    # exit()
    return X.apply(lambda x: sentence_to_vec_rep(x, word_2_vec, pad_to=max_len))


def sentence_to_elmo_vec_rep(x, word_2_vec, embedding_size=1024, pad_to=None):
    # global ALL_INV, ALL_OOV

    sentence = str(x).lower()

    if pad_to is None:
        pad_to = len(sentence.split())
    sentence_emb = np.zeros((pad_to, embedding_size), dtype=np.int64)
    idx = 0
    if len(sentence.split()) == 0:
        print('what', sentence)
        exit()

    elmo_embeds = elmo_util.embed(sentence)

    for idx in range(len(sentence.split())):
        sentence_emb[idx] = elmo_embeds[idx]

    # sentence = preprocess_sentence(sentence)
    # for word in sentence.split():
    #     # if word == 'alexa':
    #     #     continue
    #     if word in word_2_vec:
    #         sentence_emb[idx] = word_2_vec[word]
    #     else:
    #         sentence_emb[idx] = word_2_vec['UNK']
    #     idx += 1
    return sentence_emb


def to_elmo_vec_rep(X, word_2_vec):
    sentence_lens = X.apply(lambda x: len(x.split()))
    max_len = sentence_lens.max()
    print(max_len)
    # exit()
    return X.apply(lambda x: sentence_to_elmo_vec_rep(x, word_2_vec, pad_to=max_len))


def to_one_hot_vec(x, X_unique):
    # vec = np.zeros(len(X_unique))
    idx = np.where(X_unique == x)
    # vec[idx] = 1
    return idx


def to_one_hot(X):
    print('to one hot')
    X_unique = X.unique()
    print(X_unique, len(X_unique))
    return X.apply(lambda x: to_one_hot_vec(x, X_unique)), X_unique


def get_torch_dataset_loader(X, Y, batch_size=1000):
    # X, Y = get_npy_dataset(dataset_index_list, word_2_embedding)

    print(X.shape)
    print(Y.shape)

    tensor_x = torch.stack([torch.Tensor(i) for i in X])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i).long() for i in Y])

    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your dataset
    my_loader = utils.DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return my_loader


def partition_data(X, Y):
    X_train, X_test, Y_train, Y_test =\
        sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_dev, Y_train, Y_dev =\
        sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

