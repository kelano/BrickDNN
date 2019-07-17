import torch
import torch.nn.functional as F
import simple_lstm_decoupled
import re
import numpy as np
import io


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
    sentence_emb = np.zeros((seq_len, embedding_size), dtype=np.float32)
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


def run_eval_utt_decoupled(sentence, word_2_embedding, model_name, model):
    model.eval()
    model.hidden = model.init_hidden()
    x = torch.Tensor(get_embedding_sequence_encoding(sentence, word_2_embedding))

    x = x.view(1, -1, 300)
    out = model(x)
    _, pred_label = torch.max(out.data, 1)
    dd_pred = pred_label.data[0].item() == 0
    out = F.softmax(out, dim=1)
    dd_conf = out.data[0, 0]
    print '%25s > DD Conf: %f Pred (@0.5): %s' % (model_name, dd_conf.item(), dd_pred)


# embeddings
word_2_embedding = load_vectors('./wiki-news-300d-1M-subset.vec')

model = simple_lstm_decoupled.SimpleLSTMDecoupled(
            in_size=300,
            hidden_size=150,
            out_size=2,
            batch_size=1,
            lstm_layers=1,
            use_cuda=False)

model.load_state_dict(torch.load('./SimpleLSTM_Decoupled', map_location='cpu'))

# model = model.state_dict()
#
# m_states = model['model_states']
# m_params = model['args']
# class DictToObj:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
# m_params_dict = DictToObj(**m_params)





while True:
    s = raw_input('--> ')
    run_eval_utt_decoupled(s, word_2_embedding, model.name(), model)
