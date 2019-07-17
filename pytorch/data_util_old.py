import ast
import io
import os
import csv
import numpy as np
import torch
import torch.utils.data as utils
import re
from subprocess import call
import dd_platform
import os.path


EMBEDDING_MAT_CACHE = {}
WORD_2_IDX_CACHE = {}
WORD_2_VEC_CACHE = {}


class ASRRec:
    """
    Helper struct for ASR 1-best
    """
    def __init__(self, rec_str):
        rec_list = ast.literal_eval(rec_str)
        try:
            self.token_sequence = ' '.join([token[0] for token in rec_list])
        except TypeError:
            print rec_str, rec_list, type(rec_str)
            exit()
        self.token_confidences = [float(token[1]) for token in rec_list]

    def sentence(self):
        return self.token_sequence

    def confidences(self):
        return self.token_confidences


def load_embedding_as_numpy_mat(fname):
    global WORD_2_IDX_CACHE
    global EMBEDDING_MAT_CACHE

    if fname.startswith("s3://"):
        fname = download_from_s3(fname)

    if fname in WORD_2_IDX_CACHE and fname in EMBEDDING_MAT_CACHE:
        return EMBEDDING_MAT_CACHE[fname], WORD_2_IDX_CACHE[fname]

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    mat = np.zeros((n, d))
    word_2_idx = {}
    update = n/10
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        mat[count] = np.array(map(float, tokens[1:]))
        word_2_idx[tokens[0]] = count

        if count % update == 0:
            print int(float(count) / n * 100), "% complete"
        count += 1

    WORD_2_IDX_CACHE[fname] = word_2_idx
    EMBEDDING_MAT_CACHE[fname] = mat

    return mat, word_2_idx

    
def load_embedding_as_dict(fname):
    global WORD_2_VEC_CACHE

    if fname.startswith("s3://"):
        fname = download_from_s3(fname)

    if fname in WORD_2_VEC_CACHE:
        return WORD_2_VEC_CACHE[fname]

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    update = n/10
    count = 0
    word_2_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word_2_vec[tokens[0]] = map(float, tokens[1:])
        if count % update == 0:
            print int(float(count) / n * 100), "% complete"
        count += 1

    WORD_2_VEC_CACHE[fname] = word_2_vec

    return word_2_vec


def load_posteriors_file(result_file_list):
    utt_2_conf, utt_2_truth = {}, {}
    for result_file in result_file_list:
        first = True

        if result_file.startswith("s3://"):
            result_file = download_from_s3(result_file)

        with open(result_file, 'rb') as csvfile:
            results_reader = csv.reader(csvfile)
            for row in results_reader:
                if first:
                    first = False
                    continue
                utt_2_conf[row[0]] = float(row[1])
                utt_2_truth[row[0]] = int(row[2])
    return utt_2_conf, utt_2_truth


def write_posteriors_file(fname, utt_2_conf, utt_2_truth):
    with open(fname, 'wb') as csvfile:
        results_writer = csv.writer(csvfile)
        results_writer.writerow(['uttid', 'directedness_score', 'directedness_target'])
        for key in utt_2_conf.keys():
            results_writer.writerow([key, utt_2_conf[key], (1 if utt_2_truth[key] else 0)])


def download_from_s3(s3path, filename=None, download_dir=dd_platform.DATA_LOC):
    # default filename to string-ified s3 path
    if filename is None:
        filename = '_'.join([str(x) for x in s3path.split('/')[2:]])

    # create download dir if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = download_dir + '/' + filename
    print '=== downloading ' + s3path

    # already downloaded?
    if os.path.isfile(download_path):
        print '=== file already downloaded'
    else:
        print '==> s3cat.rb ' + s3path + ' > ' + download_path
        call('s3cat.rb ' + s3path + ' > ' + download_path, shell=True)
        print '<== download done'
    return download_path


def upload_to_s3(path, filename):
    # print('s3put.py -b ' + ''.join(path.split('/')[2:3]) + ' -k ' + '/'.join(path.split('/')[3:]) + filename + ' ' + filename)
    call('s3put.py -b ' + ''.join(path.split('/')[2:3]) + ' -k ' + '/'.join(
        path.split('/')[3:]) + filename + ' ' + filename, shell=True)


def get_index_metadata_col(metadata_cols, possible_labels):
    for label in possible_labels:
        if label in metadata_cols:
            return metadata_cols.index(label)
    print 'ERROR: None of column labels [%s] found' % ','.join(possible_labels)
    exit()


def get_hover_datasets(index_file_list):
    """
    returns dataset dictionaries for specified index files
    :param index_file_list: list of index files
    :return:
        utt_2_intent: utterance to NLU intent
        utt_2_sentence: utterance to ASR transcription
        utt_2_confidences: utterance to ASR token confidences
        utt_2_truth: utterance to truth (1 for DD 0 for NDD)
    """
    utt_id_labels = ['# segment..utteranceId','uttId']
    asr_result_labels = ['segment..recognition..nbest0','asr_result']
    nlu_intent_labels = ['segment..nlu1best..intent', 'segment..annotation..intent','intent']
    nd_only_labels = ['segment..transcription..isNDonly', 'segment..isNDonly','isNDOnly']

    utt_2_intent = {}
    utt_2_sentence = {}
    utt_2_confidences = {}
    utt_2_truth = {}

    for index_file in index_file_list:
        utt_id_col = None
        nlu_intent_col = None
        asr_result_col = None
        nd_only_col = None
        first = True

        if index_file.startswith('s3://'):
            index_file = download_from_s3(index_file)

        with open(index_file, 'r') as index:
            for line in index:
                splits = line.rstrip().split('\t')
                if first:
                    first = False
                    utt_id_col = get_index_metadata_col(splits, utt_id_labels)
                    asr_result_col = get_index_metadata_col(splits, asr_result_labels)
                    nlu_intent_col = get_index_metadata_col(splits, nlu_intent_labels)
                    nd_only_col = get_index_metadata_col(splits, nd_only_labels)
                    continue

                # if splits[asr_result_col].strip() == 'None' or splits[asr_result_col].strip() == '':
                #     continue

                # there have been rogue header cols in the middle of datasets
                if line.startswith("#"):
                    continue

                uttid = splits[utt_id_col].strip()
                nlu_intent = splits[nlu_intent_col].strip()
                asr_rec = ASRRec(splits[asr_result_col].strip())
                truth = splits[nd_only_col].strip()

                utt_2_intent[uttid] = nlu_intent
                utt_2_sentence[uttid] = asr_rec.sentence()
                utt_2_confidences[uttid] = asr_rec.confidences()
                utt_2_truth[uttid] = 1 if truth == 'False' else 0

                # print uttid, nlu_intent, asr_rec.sentence(), asr_rec.confidences(), truth

    return utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth


def get_npy_dataset(dataset_index_list, word_2_embedding):
    _, test_utt_2_sentence, _, test_utt_2_truth = get_hover_datasets(dataset_index_list)
    X = []
    Y = []
    empty_utts_filtered = 0
    for key in test_utt_2_sentence.keys():  # [0:max]:
        if test_utt_2_sentence[key] == "":
            # print 'what0'
            empty_utts_filtered += 1
            continue
        X.append(get_bow_encoding(test_utt_2_sentence[key], word_2_embedding))
        Y.append((int(test_utt_2_truth[key]), not int(test_utt_2_truth[key])))
    X = np.array(X)
    Y = np.array(Y)
    print '=== filtered %d empty utterances' % empty_utts_filtered
    return X, Y


def get_torch_dataset_loader(dataset_index_list, word_2_embedding, batch_size=1000):
    X, Y = get_npy_dataset(dataset_index_list, word_2_embedding)

    print X.shape
    print Y.shape

    tensor_x = torch.stack([torch.Tensor(i) for i in X])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i).long() for i in Y])

    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your dataset
    my_loader = utils.DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True)
    return my_loader


def get_npy_sequence_dataset(dataset_index_list, word_2_idx):
    _, test_utt_2_sentence, _, test_utt_2_truth = get_hover_datasets(dataset_index_list)

    pad_size = max([len(test_utt_2_sentence[uttid].split()) for uttid in test_utt_2_sentence.keys()])

    X = []
    Y = []
    for key in test_utt_2_sentence.keys():  # [0:max]:
        if test_utt_2_sentence[key] == "":
            # print 'what0'
            continue
        X.append(get_sequence_encoding(test_utt_2_sentence[key], word_2_idx, pad_to=pad_size))
        # Y.append((int(test_utt_2_truth[key] == 'True'), int(test_utt_2_truth[key] != 'True')))
        Y.append((int(test_utt_2_truth[key]), int(not test_utt_2_truth[key])))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def get_torch_sequence_dataset_loader(dataset_index_list, word_2_idx, batch_size=1000):
    X, Y = get_npy_sequence_dataset(dataset_index_list, word_2_idx)

    print X.shape
    print Y.shape

    # tensor_x = torch.stack([torch.Tensor(i) for i in X])  # transform to torch tensors
    tensor_x = torch.Tensor(X).long()  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i).long() for i in Y])

    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your dataset
    # drop last batch as we require exactly 'batch_size' items for our recurrent models
    my_loader = utils.DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return my_loader



ALL_INV = 0
ALL_OOV = 0
OOV_DICT = {}
def get_bow_encoding(sentence, word_2_embedding, embedding_size=300):
    global ALL_INV, ALL_OOV
    sentence_emb = np.zeros(embedding_size)
    if len(sentence.split()) == 0:
        print 'what', sentence
        exit()
    sentence = preprocess_sentence(sentence)
    for word in sentence.split():
        # if word == 'alexa':
        #     continue
        if word in word_2_embedding:
            word_emb = np.array(word_2_embedding[word])
            ALL_INV += 1
        else:
            word_emb = np.array(word_2_embedding['unk'])
            ALL_OOV += 1
            # print word
            if word not in OOV_DICT:
                OOV_DICT[word] = 1
            else:
                OOV_DICT[word] = OOV_DICT[word] + 1
        sentence_emb += word_emb
    sentence_emb /= len(sentence.split())
    return sentence_emb


def get_sequence_encoding(sentence, word_2_idx, pad_to=None):
    global ALL_INV, ALL_OOV

    if pad_to is None:
        pad_to = len(sentence.split())
    sentence_emb = np.ones(pad_to, dtype=np.int64) * word_2_idx['<PAD>']
    idx = 0
    if len(sentence.split()) == 0:
        print 'what', sentence
        exit()
    sentence = preprocess_sentence(sentence)
    for word in sentence.split():
        # if word == 'alexa':
        #     continue
        if word in word_2_idx:
            sentence_emb[idx] = word_2_idx[word]
            ALL_INV += 1
        else:
            sentence_emb[idx] = word_2_idx['unk']
            ALL_OOV += 1
            # print word
            if word not in OOV_DICT:
                OOV_DICT[word] = 1
            else:
                OOV_DICT[word] = OOV_DICT[word] + 1
        idx += 1
    return sentence_emb


def get_embedding_sequence_encoding(sentence, word_2_embedding, embedding_size=300, pad_to=None):
    global ALL_INV, ALL_OOV

    if pad_to is None:
        pad_to = len(sentence.split())
    sentence_emb = np.ones((pad_to, embedding_size), dtype=np.int64) * word_2_embedding['<PAD>']
    idx = 0
    if len(sentence.split()) == 0:
        print 'what', sentence
        exit()
    sentence = preprocess_sentence(sentence)
    for word in sentence.split():
        # if word == 'alexa':
        #     continue
        if word in word_2_embedding:
            sentence_emb[idx] = word_2_embedding[word]
            ALL_INV += 1
        else:
            sentence_emb[idx] = word_2_embedding['unk']
            ALL_OOV += 1
            # print word
            if word not in OOV_DICT:
                OOV_DICT[word] = 1
            else:
                OOV_DICT[word] = OOV_DICT[word] + 1
        idx += 1
    return sentence_emb


def preprocess_sentence(sentence):
    """
    Preprocessing for input sentences. Currently only removes punctuation and non-alphanumeric characters
    :param sentence:
    :return:
    """
    # remove punctuation
    return re.sub(ur"[^\w\d\s]+", '', sentence)


def intersection(d1, d2):
    """
    Returns an intersection set of the two input sets
    :param d1:
    :param d2:
    :return:
    """
    dint = {}
    for key in d1.keys():
        if key in d2:
            dint[key] = d1[key]
    return dint


def common_filter(d1, d2):
    """
    Removes items from input sets that are not part of the intersection
    :param d1:
    :param d2:
    :return:
    """
    keys_a = set(d1.keys())
    keys_b = set(d2.keys())
    int_uttids = keys_a & keys_b

    for k, v in d1.items():
        if k not in int_uttids:
            del(d1[k])
            
    for k, v in d2.items():
        if k not in int_uttids:
            del(d2[k])
    
    return d1, d2