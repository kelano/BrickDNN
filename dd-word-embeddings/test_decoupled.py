import data_util
import torch
import torch.nn.functional as F
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

def run_eval_utt_decoupled(sentence, word_2_embedding, model_name, model):
    model.eval()
    model.hidden = model.init_hidden()
    x = torch.Tensor(data_util.get_embedding_sequence_encoding(sentence, word_2_embedding))

    x = x.view(1, -1, 300)
    out = model(x)
    _, pred_label = torch.max(out.data, 1)
    dd_pred = pred_label.data[0].item() == 0
    out = F.softmax(out, dim=1)
    dd_conf = out.data[0, 0]
    print '%25s > DD Conf: %f Pred (@0.5): %s' % (model_name, dd_conf.item(), dd_pred)



data_loc = dd_platform.DATA_LOC

import trained_model_groups
import torch


# embeddings
embedding = trained_model_groups.models['Prod.v100.Local']['SimpleLSTM_150H']['embedding']
# embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec'
embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat(embedding)
word_2_embedding = data_util.load_embedding_as_dict(embedding)

config_overrides = {"BATCH_SIZE": 1}  # Override for single element batches for forward pass

models = []
models.append(('Coupled', data_util.load_trained_model('Prod.v100.Local', 'SimpleLSTM_150H', embedding_mat=embedding_mat, config_overrides=config_overrides)))
models.append(('Decoupled', data_util.load_trained_model('Prod.v100.Local', 'SimpleLSTM_150H_Decoupled', embedding_mat=embedding_mat, config_overrides=config_overrides)))



#
# # model_path = 'BOW-MLP'
# model_path = 'SimpleLSTM_150H_TL'
# model_name = model_path.split('/')[-1]
# from configs.simple_lstm_150H_TL import *
# model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
#
# models.append((model_name, model))
#
# # model_path = 'BOW-MLP'
# model_path = 'SimpleLSTM_150H_TL_FT'
# model_name = model_path.split('/')[-1]
# from configs.simple_lstm_150H_TL_FineTune import *
# model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
#
# models.append((model_name, model))
#
# model_path = 'SimpleLSTM_150H_05005'
# model_name = model_path.split('/')[-1]
# from configs.simple_lstm_150H import *
# model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, use_cuda=False)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
#
# models.append((model_name, model))
#
# model_path = 'SimpleLSTM_150H_2Stack'
# model_name = model_path.split('/')[-1]
# from configs.simple_lstm_2stack_150H import *
# model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, lstm_layers=LSTM_LAYERS, use_cuda=False)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
#
# model.summary()
# exit()
#
# models.append((model_name, model))

coupled_tup = models[0]
decoutpled_tup = models[1]
while True:
    s = raw_input('--> ')
    run_eval_utt(s, word_2_idx, coupled_tup[0], coupled_tup[1])
    run_eval_utt_decoupled(s, word_2_embedding, decoutpled_tup[0], decoutpled_tup[1])
