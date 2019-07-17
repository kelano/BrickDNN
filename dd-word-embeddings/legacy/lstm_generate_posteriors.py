# -*- coding: utf-8 -*-

import data_util
import torch
import torch.nn.functional as F
from models.simple_lstm import SimpleLSTM
import dd_platform
import time


ALL_MISSES = dict()


def run_eval_utt(utt_2_sentence, word_2_idx, model, utt_2_truth, utt_2_intent):
    global ALL_MISSES
    model.eval()
    utt_2_conf = {}
    update_freq = len(utt_2_sentence) / 100
    print update_freq
    count = 0
    for key in utt_2_sentence.keys():
        # total = time.time()
        model.hidden = model.init_hidden()

        sentence = utt_2_sentence[key]
        if sentence == '':
            utt_2_conf[key] = 1.0
            continue


        # start = time.time()
        x = torch.Tensor(data_util.get_sequence_encoding(sentence, word_2_idx, pad_to=len(sentence.split()))).long()
        # print 'encoding', time.time() - start

        # wrap as a 1-length batch
        x = x.view(1, -1)

        # start = time.time()
        out = model(x)
        # print 'forward pass', time.time() - start

        # start = time.time()
        # _, pred_label = torch.max(out.data, 1)
        # dd_truth = utt_2_truth[key] == 'False'
        # dd_pred = pred_label.data[0].item() == 0
        
        # if dd_truth != dd_pred:
        #     dk = '%s-%s-%s-%s' % (sentence, utt_2_intent[key], dd_pred, dd_truth)
        #     if dk not in ALL_MISSES:
        #         ALL_MISSES[dk] = 1
        #     else:
        #         ALL_MISSES[dk] = ALL_MISSES[dk] + 1
        
        out = F.softmax(out, dim=1)
        utt_2_conf[key] = out.data[0, 1].item()

        if count % update_freq == 0:
            print float(count) / len(utt_2_sentence) * 100,'%'
        count += 1
        # print 'rest', time.time() - start
        # print 'total', time.time() - total
        # exit()
    return utt_2_conf



# model = BOWMLP(300, 2, 600)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))


# if use_cuda:
#     model = model.cuda()

data_loc = dd_platform.DATA_LOC

# data_type = 'prodv1'
# dataset_name = 'test'
data_type = 'ADS'
dataset_name = 'test.ADS.with_WW.with_Empty.Week43-44'
# dataset_name = 'test.ADS.Week43-44'
# data_type = 'mixed'
# dataset_name = 'test.prodv1_ADS.Week43-44'

index = '%s/%s/%s.index' % (data_loc, data_type, dataset_name)
results_pre = '%s/%s/%s.%s-results.csv' % (data_loc, data_type, dataset_name, '%s')

# word_2_embedding = data_util.load_vectors('%s/embeddings/wiki-news-300d-1M-subset.vec' % data_loc)

embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat('%s/embeddings/wiki-news-300d-1M-subset.vec' % data_loc)



# model_path = 'BOW-MLP'
# model_path = 'BOW-MLP-MixedTrain'
# model_path = 'SimpleLSTM_TEST'

# model_path = 'SimpleLSTM_150H'
from configs.simple_lstm_150H import *

# model_path = 'SimpleLSTM_150H_1PAT'

# model_path = 'SimpleLSTM_150H_2Stack'
# from configs.simple_lstm_2stack_150H import *

model_path = '/Users/kelleng/Desktop/trained-models/SimpleLSTM_150H_1PAT'
# from configs.simple_lstm_150H import *
#
# model_path = 'SimpleLSTM_150H_TL'
# from configs.simple_lstm_150H_TL import *

# model_path = 'SimpleLSTM_150H_TL_FT'
# from configs.simple_lstm_150H_TL_FineTune import *

model_name = model_path.split('/')[-1]
model = SimpleLSTM(embedding_mat=embedding_mat, hidden_size=HIDDEN_SIZE, out_size=2, batch_size=1, lstm_layers=LSTM_LAYERS, use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))


# from torchsummary import summary
# summary(model, [(5)])

# model.summary()
# model.eval()
# model.summary()
# model.train()
# model.summary()
# exit()

utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_dataset(index)

s2_utt_2_conf = run_eval_utt(utt_2_sentence, word_2_idx, model, utt_2_truth, utt_2_intent)
print data_util.ALL_INV, data_util.ALL_OOV
data_util.write_dory_results(results_pre % model_name, s2_utt_2_conf, utt_2_truth)

# sd = sorted(data_util.OOV_DICT.items(), key=lambda i: i[1])
# sd = sorted(ALL_MISSES.items(), key=lambda i: i[1])
#
# print sd[-10:]