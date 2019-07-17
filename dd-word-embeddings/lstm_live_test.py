import data_util
import torch
import torch.nn.functional as F
from models.simple_lstm import SimpleLSTM
import dd_platform


def run_eval_utt(sentence, word_2_idx, model_name, model):
    model.eval()
    model.hidden = model.init_hidden()
    x = torch.Tensor(data_util.get_sequence_encoding(sentence, word_2_idx)).long()

    x = x.view(1, -1)
    out = model(x)
    _, pred_label = torch.max(out.data, 1)
    dd_pred = pred_label.data[0].item() == 0
    out = F.softmax(out, dim=1)
    dd_conf = out.data[0, 0]
    print '%25s > DD Conf: %f Pred (@0.5): %s' % (model_name, dd_conf.item(), dd_pred)


data_loc = dd_platform.DATA_LOC
embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat('%s/embeddings/wiki-news-300d-1M-subset.vec' % data_loc)

models = []

# model_path = 'BOW-MLP'
model_path = 'SimpleLSTM_150H_TL'
model_name = model_path.split('/')[-1]
from configs.simple_lstm_150H_TL import *
model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

models.append((model_name, model))

# model_path = 'BOW-MLP'
model_path = 'SimpleLSTM_150H_TL_FT'
model_name = model_path.split('/')[-1]
from configs.simple_lstm_150H_TL_FineTune import *
model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

models.append((model_name, model))

model_path = 'SimpleLSTM_150H_05005'
model_name = model_path.split('/')[-1]
from configs.simple_lstm_150H import *
model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

models.append((model_name, model))

model_path = 'SimpleLSTM_150H_2Stack'
model_name = model_path.split('/')[-1]
from configs.simple_lstm_2stack_150H import *
model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, lstm_layers=LSTM_LAYERS, use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

model.summary()
exit()

models.append((model_name, model))

while True:
    s = raw_input('--> ')
    for model_tup in models:
        run_eval_utt(s, word_2_idx, model_tup[0], model_tup[1])
