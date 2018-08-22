import io
import numpy as np


def load_vectors():
    fin = io.open('wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(x) for x in tokens[1:]])
        if count % 100000 == 0:
            print(int(float(count) / 1000000 * 100),"% complete")
        count = count + 1
    return data


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
