# -*- coding: utf-8 -*-

import data_util
import torch
import torch.nn.functional as F
from models.baseline_bow_mlp import BOWMLP
import dd_platform


ALL_MISSES = dict()


def run_eval_utt(utt_2_sentence, word_2_embedding, model, utt_2_truth, utt_2_intent):
    global ALL_MISSES
    model.eval()
    utt_2_conf = {}
    for key in utt_2_sentence.keys():
        sentence = utt_2_sentence[key]
        if sentence == '':
            utt_2_conf[key] = 1.0
            continue
        
        x = torch.Tensor(data_util.get_bow_encoding(sentence, word_2_embedding))
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
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
    return utt_2_conf




from models.baseline_bow_mlp import BOWMLP
from configs.bow_mlp_v1 import *

# dataset_group = 'ASI.201809-201811'
dataset_group = 'Prod.v104'
dataset_type = 'test'

import dataset_groups
datasets = dataset_groups.groups[dataset_group][dataset_type]

model_path = 'BOW-MLP'
# model_path = 'BOW-MLP-MixedTrain'
model_name = model_path.split('/')[-1]

model = BOWMLP(300, 2, 600)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

data_loc = dd_platform.DATA_LOC

# # data_type = 'prodv1'
# # dataset_name = 'test'
# data_type = 'ADS'
# # dataset_name = 'test.ADS.Week43-44'
# dataset_name = 'test.ADS.with_WW.with_Empty.Week43-44'
# # data_type = 'mixed'
# # dataset_name = 'test.prodv1_ADS.Week43-44'
#
# index = '%s/%s/%s.index' % (data_loc, data_type, dataset_name)
out_name = '%s.%s.%s-results.csv' % (dataset_group, dataset_type, model_name)

# word_2_embedding = data_util.load_vectors('%s/embeddings/wiki-news-300d-1M-subset.vec' % data_loc)
word_2_embedding = data_util.load_embedding_as_dict('s3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec')

utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(datasets)

s2_utt_2_conf = run_eval_utt(utt_2_sentence, word_2_embedding, model, utt_2_truth, utt_2_intent)
print data_util.ALL_INV, data_util.ALL_OOV
data_util.write_posteriors_file(out_name, s2_utt_2_conf, utt_2_truth)

# sd = sorted(data_util.OOV_DICT.items(), key=lambda i: i[1])
# sd = sorted(ALL_MISSES.items(), key=lambda i: i[1])
#
# print sd[-10:]