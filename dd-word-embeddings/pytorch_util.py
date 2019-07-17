import torch
import torch.nn as nn
import trained_model_groups
import data_util
import json
from models import baseline_bow_mlp, simple_lstm, simple_lstm_decoupled


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


def sort_by_sequence_length(x, batch_size, use_cuda):
    lengths = torch.Tensor([torch.nonzero(x[idx])[-1].item() + 1 for idx in range(0, batch_size)]).long()
    if use_cuda:
        lengths.cuda()

    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_x = x[sorted_indices]
    return sorted_x, sorted_lengths, sorted_indices


def sort_by_sequence_length_embedding(x, batch_size, use_cuda):
    lengths = torch.Tensor([torch.nonzero(x[idx])[-1][0].item() + 1 for idx in range(0, batch_size)]).long()
    if use_cuda:
        lengths.cuda()

    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_x = x[sorted_indices]
    return sorted_x, sorted_lengths, sorted_indices


def load_trained_model(model_group, model_name, embedding_mat=None, use_cuda=False, config_overrides={}):
    model_path = trained_model_groups.models[model_group][model_name]["loc"]
    if model_path.startswith('s3://'):
        model_path = data_util.download_from_s3(model_path)

    model, conf = create_model_instance(model_group, model_name, embedding_mat, use_cuda, config_overrides)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return model, conf


def create_model_instance(model_group, model_name, embedding_mat=None, use_cuda=False, config_overrides={}):
    # load config
    model_config_name = trained_model_groups.models[model_group][model_name]["config"]
    with open('./configs/model_configs.json', 'r') as infile:
        conf = json.load(infile)[model_config_name]

    # add config overrides
    for k, v in config_overrides.items():
        print 'Overriding config param %s to value %s' % (k, str(v))
        conf[k] = v

    # create pytorch model instance
    if 'BOW' in model_name:
        model = baseline_bow_mlp.BOWMLP(
            in_size=300,
            out_size=2,
            hidden_size=conf["HIDDEN_SIZE"])
    elif 'Decoupled' in model_name:
        model = simple_lstm_decoupled.SimpleLSTMDecoupled(
            in_size=300,
            hidden_size=conf["HIDDEN_SIZE"],
            out_size=2,
            batch_size=conf["BATCH_SIZE"],
            lstm_layers=conf["LSTM_LAYERS"],
            use_cuda=use_cuda)
    else:
        model = simple_lstm.SimpleLSTM(
            embedding_mat=embedding_mat,
            hidden_size=conf["HIDDEN_SIZE"],
            out_size=2,
            batch_size=conf["BATCH_SIZE"],
            lstm_layers=conf["LSTM_LAYERS"],
            use_cuda=use_cuda)

    # load start point if specified
    if 'START_POINT' in conf:
        start_point_model_name = conf["START_POINT"]
        start_point_model, _conf = load_trained_model(model_group, start_point_model_name, embedding_mat, use_cuda)
        model.load_state_dict(start_point_model.state_dict())

    # freeze layers if specified
    if 'FROZEN_LAYERS' in conf:
        frozen_layers = conf["FROZEN_LAYERS"]
        print frozen_layers
        for name, param in model.named_parameters():
            if name in frozen_layers:
                # print 'match!'
                param.requires_grad = False
            print name, param.requires_grad

    return model, conf

