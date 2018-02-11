import csv
import numpy as np
import string
from nltk.stem.porter import *
import unidecode


def get_bow_multi_encoding(bow_vec, str_values):
    encoding = np.zeros(len(bow_vec))
    for str_value in str_values:
        encoding[bow_vec.index(str_value)] += 1
    return encoding.tolist()


def get_bow_encoding(bow_vec, str_value, split=False, stem=False):
    stemmer = PorterStemmer()
    encoding = np.zeros(len(bow_vec))
    str_value = prep_str(str_value)
    if split:
        for item in str_value.split():
            if stem:
                item = stemmer.stem(item)
            if item in bow_vec:
                encoding[bow_vec.index(item)] += 1
            else:
                encoding[bow_vec.index('UNK')] += 1
    else:
        if stem:
            str_value = stemmer.stem(str_value)
        if str_value in bow_vec:
            encoding[bow_vec.index(str_value)] += 1
        else:
            encoding[bow_vec.index('UNK')] += 1
    return encoding.tolist()


def get_bow_vec(data, row_index, split, stem):
    bow_vec = []
    stemmer = PorterStemmer()
    for data_row in data:
        val = prep_str(str(data_row[row_index]))
        if split:
            for item in val.split():
                if stem:
                    # print item
                    item = stemmer.stem(item)
                if item.isalpha():
                    if item not in bow_vec:
                        bow_vec.append(item)
        else:
            if stem:
                val = stemmer.stem(val)
            if val not in bow_vec:
                bow_vec.append(val)
    bow_vec = sorted(bow_vec)
    bow_vec.append('UNK')
    return bow_vec


def get_bow_vec_raw(raw_data, stem=False):
    bow_vec = []
    stemmer = PorterStemmer()
    for data in raw_data:
        val = prep_str(str(data))
        if stem:
            val = stemmer.stem(val)
        if val not in bow_vec:
            bow_vec.append(val)
    bow_vec = sorted(bow_vec)
    bow_vec.append('UNK')
    return bow_vec


def prep_str(str):
    str = unidecode.unidecode(unicode(str, "utf-8"))
    return str.translate(None, string.punctuation).lower().strip()


def get_bow_vecs(col, split=False, stem=False):
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

        return get_bow_vec(data, headers.index(col), split, stem)

#
#set_name_bow_vec = get_bow_vecs('Name', split=True)
# theme_bow_vec = get_bow_vecs('Theme')
# print 'hi'