import io
import numpy as np


def load_vectors():
    fin = io.open('..\wiki-news-300d-1M-subset.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_2_vec = {}
    word_to_idx = {}
    idx_to_vec = {}
    tf_embedding = np.zeros([6047, 300])
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vec = np.array([float(x) for x in tokens[1:]])
        word_2_vec[word] = vec
        word_to_idx[word] = count
        idx_to_vec[count] = vec
        tf_embedding[count] = vec
        if count % 1000 == 0:
            print(int(float(count) / 6407 * 100),"% complete")
        count = count + 1
    return word_2_vec, word_to_idx, idx_to_vec, tf_embedding


def load_vectors_full():
    fin = io.open('..\wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_2_vec = {}
    word_to_idx = {}
    idx_to_vec = {}
    tf_embedding = np.zeros([1000000, 300])
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vec = np.array([float(x) for x in tokens[1:]])
        word_2_vec[word] = vec
        word_to_idx[word] = count
        idx_to_vec[count] = vec
        tf_embedding[count] = vec
        if count % 100000 == 0:
            print(int(float(count) / 1000000 * 100),"% complete")
        count = count + 1
    return word_2_vec, word_to_idx, idx_to_vec, tf_embedding


def get_fasttext_encoding(data, str_value):
    encoding = np.zeros(300, dtype=float)
    # print(str_value)
    for item in str_value.split():
        item = item.lower()
        if item in data:
            # print(item, data[item])
            encoding += data[item]
        else:
            # print(item, 'UNK', data['UNK'])
            encoding += data['UNK']
    # print(str_value, encoding / len(str_value.split()))
    # exit()
    return (encoding / len(str_value.split())).tolist()


def get_fasttext_tokenization(word_2_idx, str_value, pad_to=None):
    split = str_value.split()
    len_split = len(split)
    tokens = np.zeros(len_split if pad_to is None else pad_to, dtype='int32')
    for word_idx in range(0, len_split):
        word = split[word_idx].lower()
        tokens[word_idx] = word_2_idx[word] if word in word_2_idx else word_2_idx['UNK']
    return tokens.tolist()
