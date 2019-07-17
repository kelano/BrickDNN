import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import sys
sys.path.append('..')
import pytorch_util


class SimpleLSTMDecoupled(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, batch_size, lstm_layers=1, use_cuda=False):
        super(SimpleLSTMDecoupled, self).__init__()
        self.use_cuda = use_cuda

        self.in_size = in_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm_layers = lstm_layers

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(in_size, hidden_size, lstm_layers)

        # The linear layer that maps from hidden state space to tag space
        self.aff = nn.Linear(hidden_size, out_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size)

        if self.use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()

        return (h0, c0)


    def forward(self, x):
        # sort by length for packing
        sorted_x, sorted_lengths, sorted_indices = pytorch_util.sort_by_sequence_length_embedding(x, self.batch_size, self.use_cuda)

        # encode
        # embeds = self.word_embeddings(sorted_x)

        # pack
        packed = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lengths, batch_first=True)

        # lstm forward pass
        # start = time.time()
        lstm_outs, self.hidden = self.lstm(packed, self.hidden)
        # print 'internal fwd', time.time() - start

        # uncomment me to use or view raw output
        # lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs, batch_first=True)

        # linear
        # aff_out = F.relu(self.aff(self.hidden[0][-1]))
        aff_out = self.aff(self.hidden[0][-1]) # no ReLU pre-softmax

        # unsort output to match original x order for loss calculation
        _unsorted, unsorted_indices = torch.sort(sorted_indices, descending=False)
        unsorted_output = aff_out[unsorted_indices, :]

        return unsorted_output


    def summary(self):
        for layer in [self.word_embeddings, self.lstm, self.aff]:
            print '-->>> Layer: %30s\tTraining: %s' % (layer, layer.training)
            for param in layer.named_parameters():
                pstr = ''
                for i in range(0, len(param)):
                    if i != 1:
                        pstr = '%s\t%15s' % (pstr, param[i])
                    else:
                        pstr = '%s\tRequires_Grad: %s' % (pstr, param[i].requires_grad)
                print pstr


    def name(self):
        return "SimpleLSTM_TEST"
