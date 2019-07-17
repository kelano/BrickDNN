# -*- coding: utf-8 -*-

"""
Utility for running forward passes for specified models and datasets to generate posteriors files to be used in eval
scripts and other tools.
"""

import data_util
import torch
import torch.nn.functional as F
import dd_platform
import os
import trained_model_groups
import dataset_groups
import pytorch_util


def run_forward_pass(utt_2_sentence, word_2_embedding, word_2_idx, model, model_group, model_name):
    # empty utterances default to non-DD
    empty_utt_dd_score = 0.0

    # we switched dd_class index from 0 to 1 after v100
    # TODO: remove me once we no longer need v100 data
    dd_class_index = 0 if model_group == 'Prod.v100' else 1

    # put model in eval mode
    model.eval()

    utt_2_conf = {}
    update_freq = len(utt_2_sentence) / 10
    count = 0
    for key in utt_2_sentence.keys():
        sentence = utt_2_sentence[key]
        if sentence == '':
            utt_2_conf[key] = empty_utt_dd_score
            continue

        if 'MLP' in model_name:
            # get BOW embedding encoding
            x = torch.Tensor(data_util.get_bow_encoding(sentence, word_2_embedding))
        else:
            # get sequence word index encoding (embedding is part of model in this case)
            x = torch.Tensor(data_util.get_sequence_encoding(sentence, word_2_idx, pad_to=len(sentence.split()))).long()
            # wrap as batch of 1
            x = x.view(1, -1)
            # init hidden state
            model.hidden = model.init_hidden()

        # run forward pass and get posterior
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        out = F.softmax(out, dim=1)
        utt_2_conf[key] = out.data[0, dd_class_index].item()

        if count % update_freq == 0:
            print float(count) / len(utt_2_sentence) * 100, '%'
        count += 1
    return utt_2_conf


def generate_posteriors(model_group, model_name, dataset_group, dataset_name):
    # load if it's there!
    local_file = '%s/%s.%s.%s.%s-results.csv' % (dd_platform.DATA_LOC, dataset_group, dataset_name, model_group, model_name)
    if os.path.isfile(local_file):
        print '=== loading local posteriors ' + local_file
        utt_2_conf, utt_2_truth = data_util.load_posteriors_file([local_file])
        return utt_2_conf, utt_2_truth

    print '=== generating posteriors ' + local_file

    # embeddings
    embedding = trained_model_groups.models[model_group][model_name]['embedding']
    embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat(embedding)
    word_2_embedding = data_util.load_embedding_as_dict(embedding)

    # misc
    use_cuda = False

    # load model
    config_overrides = { "BATCH_SIZE": 1 } # Override for single element batches for forward pass
    model, conf = pytorch_util.load_trained_model(model_group, model_name, embedding_mat, use_cuda, config_overrides)
    datasets = dataset_groups.groups[dataset_group][dataset_name]

    # run forward pass
    utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(datasets)
    s2_utt_2_conf = run_forward_pass(utt_2_sentence, word_2_embedding, word_2_idx, model, model_group, model_name)
    data_util.write_posteriors_file(local_file, s2_utt_2_conf, utt_2_truth)

    return s2_utt_2_conf, utt_2_truth


def main():
    """
    Generates posteriors for all dataset model combinations by default (this takes a while). Modify the dataset group
    and/or model group to manually generate a subset of posteriors.
    """
    for dataset_group in dataset_groups.groups.keys():
        dataset = dataset_groups.groups[dataset_group]
        if 'test' in dataset:
            dataset_name = 'test'
            for model_group in trained_model_groups.models.keys():
                for model_name in trained_model_groups.models[model_group].keys():
                    print dataset_group, dataset_name, model_group, model_name
                    generate_posteriors(model_group, model_name, dataset_group, dataset_name)


if __name__ == "__main__":
    main()
