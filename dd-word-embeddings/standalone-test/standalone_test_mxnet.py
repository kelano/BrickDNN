import re
import numpy as np
import io
import mxnet as mx


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

    seq_len = len(sentence.split())
    if seq_len == 0:
        print '=== exiting'
        exit()
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


def run_eval_utt_decoupled(sentence, word_2_embedding, model_name, mod):
    x = mx.nd.array(get_embedding_sequence_encoding(sentence, word_2_embedding, pad_to=10))
    x = mx.nd.reshape(x, (1, 10, 300))

    # mod.forward(x)
    mod.forward(Batch([x]))
    out = mod.get_outputs()[0].asnumpy() # (1 10 2)
    out = out.squeeze() # (10 2)

    seq_len = len(sentence.split())
    dd_conf = out[seq_len - 1, 0]
    dd_pred = dd_conf > 0.5

    print '%25s > DD Conf: %f Pred (@0.5): %s' % (model_name, dd_conf.item(), dd_pred)


ctx = mx.cpu()

# mod = mx.gluon.nn.SymbolBlock.imports("test_mxnet-symbol.json", ['data'], "test_mxnet-0000.params", ctx=ctx)
#
sym, arg_params, aux_params = mx.model.load_checkpoint('test_mxnet', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,10,300))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# embeddings
# word_2_embedding = load_vectors('./wiki-news-300d-1M-subset.vec')
word_2_embedding = load_vectors('/Users/kelleng/data/dd/embeddings/wiki-news-300d-1M-subset.vec')


while True:
    s = raw_input('--> ')
    run_eval_utt_decoupled(s, word_2_embedding, 'mxnet', mod)
