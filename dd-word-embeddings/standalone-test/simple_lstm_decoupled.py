import torch
import torch.nn as nn


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
        x = x.view(x.shape[1], 1, -1)
        lstm_outs, self.hidden = self.lstm(x, self.hidden)
        aff_out = self.aff(self.hidden[0][-1]) # no ReLU pre-softmax
        return aff_out


    def summary(self):
        for layer in [self.lstm, self.aff]:
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
        return "SimpleLSTM_Decoupled"
