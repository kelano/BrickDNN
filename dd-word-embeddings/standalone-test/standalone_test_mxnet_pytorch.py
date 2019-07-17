import re
import numpy as np
import io
import datetime

import torch
import torch.nn.functional as F
import simple_lstm_decoupled

import mxnet as mx
from collections import namedtuple

UNK_TOKEN = 'unk'


def load_vectors(fname):
    print '=== loading embedding'
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    update = n/10
    count = 0
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        if count % update == 0:
            print int(float(count) / n * 100), "% complete"
        count += 1
    return data


def preprocess_sentence(sentence):
    # remove punctuation
    return re.sub(ur"[^\w\d\s]+", '', sentence)


def get_embedding_sequence_encoding(sentence, word_2_embedding, embedding_size=300, pad_to=None):
    global UNK_TOKEN

    if pad_to is None:
        seq_len = len(sentence.split())
        if seq_len == 0:
            print '=== exiting'
            exit()
        pad_to = seq_len

    sentence_emb = np.zeros((pad_to, embedding_size), dtype=np.float32)
    idx = 0
    sentence = preprocess_sentence(sentence)
    for word in sentence.split():
        # if word == 'alexa':
        #     continue
        if word in word_2_embedding:
            sentence_emb[idx] = word_2_embedding[word]
        else:
            sentence_emb[idx] = word_2_embedding[UNK_TOKEN]
        idx += 1
    return sentence_emb


def load_pytorch():
    model = simple_lstm_decoupled.SimpleLSTMDecoupled(
        in_size=300,
        hidden_size=150,
        out_size=2,
        batch_size=1,
        lstm_layers=1,
        use_cuda=False)

    model.load_state_dict(torch.load('./SimpleLSTM_Decoupled', map_location='cpu'))
    return model


def load_mxnet():
    ctx = mx.cpu()

    # mod = mx.gluon.nn.SymbolBlock.imports("test_mxnet-symbol.json", ['data'], "test_mxnet-0000.params", ctx=ctx)
    #
    sym, arg_params, aux_params = mx.model.load_checkpoint('test_mxnet', 0)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1,10,300))], label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def pytorch_forward(sentence, word_2_embedding, model_name, model):
    model.eval()
    x = torch.Tensor(get_embedding_sequence_encoding(sentence, word_2_embedding))

    x = x.view(1, -1, 300)
    s = datetime.datetime.now()
    model.hidden = model.init_hidden()
    out = model(x)
    _, pred_label = torch.max(out.data, 1)
    dd_pred = pred_label.data[0].item() == 0
    out = F.softmax(out, dim=1)
    dd_conf = out.data[0, 0]
    e = datetime.datetime.now()
    print '%25s > DD Conf: %f\tPred (@0.5): %s\tTime: %d' % (model_name, dd_conf.item(), dd_pred, int((e-s).total_seconds()*1000))


def mxnet_forward(sentence, word_2_embedding, model_name, model):
    Batch = namedtuple('Batch', ['data'])

    x = mx.nd.array(get_embedding_sequence_encoding(sentence, word_2_embedding, pad_to=10))
    x = mx.nd.reshape(x, (1, 10, 300))

    # mod.forward(x)
    s = datetime.datetime.now()
    model.forward(Batch([x]))
    out = model.get_outputs()[0].asnumpy() # (1 10 2)
    out = out.squeeze() # (10 2)

    seq_len = len(sentence.split())
    dd_conf = out[seq_len - 1, 0]
    dd_pred = dd_conf > 0.5
    e = datetime.datetime.now()

    print '%25s > DD Conf: %f\tPred (@0.5): %s\tTime: %d' % (model_name, dd_conf.item(), dd_pred, int((e-s).total_seconds()*1000))


# embeddings
word_2_embedding = load_vectors('./wiki-news-300d-1M-subset.vec')
pymodel = load_pytorch()
mxmodel = load_mxnet()


while True:
    s = raw_input('--> ')
    pytorch_forward(s, word_2_embedding, 'pytorch', pymodel)
    mxnet_forward(s, word_2_embedding, 'mxnet', mxmodel)
