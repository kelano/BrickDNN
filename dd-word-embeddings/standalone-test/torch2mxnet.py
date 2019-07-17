from __future__ import  print_function, division
import sys, os, math, datetime, random

import numpy as np

import simple_lstm_decoupled

import torch
import torch.autograd as autograd

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn as nn
from mxnet.gluon import rnn as rnn
from mxnet.gluon.parameter import Parameter as mxparam

use_cuda = torch.cuda.is_available()
mxctx = mx.gpu() if use_cuda else mx.cpu()

def masked_softamx(F, x, mask, dim=1):
    '''
        This function does masked-softmax
        inputs:
            x must be [b, seq_len, 1]
            mask must be [b, seq_length]
        returns:
            [b, seq_len, 1]

    '''
    x = F.squeeze(x, axis=-1) # [b, seq_len, 1] ==> [b, seq_len]
    x = mask * x # [b, seq_len]
    x = F.broadcast_div(x, (F.sum(x, axis=dim, keepdims=True) + 1e-13))
    x = F.expand_dims(x, axis=-1) # [b, seq_len] ==> [b, seq_len, 1]

    return x

class SimpleLSTM_Decoupled_mxnet(gluon.HybridBlock):
    def __init__(self, in_size, hidden_size, out_size, batch_size, lstm_layers=1, **kwargs):
        '''
            Simple baseline model:
            This model use a Fast text word embedding + simple linear layers
        '''
        super(SimpleLSTM_Decoupled_mxnet, self).__init__(**kwargs)

        with self.name_scope():
            self.in_size = in_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.out_size = out_size
            self.lstm_layers = lstm_layers

            self.lstm = rnn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.lstm_layers, layout='NTC')
            self.aff = nn.Dense(units=self.out_size, in_units=self.hidden_size, use_bias=True, flatten=False)

            # self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)



        h0 = mx.sym.zeros(shape=(self.lstm_layers, self.batch_size, self.hidden_size))
        c0 = mx.sym.zeros(shape=(self.lstm_layers, self.batch_size, self.hidden_size))

        # if self.use_cuda:
        #     h0, c0 = h0.cuda(), c0.cuda()

        return [h0, c0]

    def hybrid_forward(self, F, x):
        # self.hidden = self.lstm.begin_state(self.batch_size, func=mx.sym.zeros)

        # 7. Get the scores
        # output, hn = self.lstm(x, self.hidden)
        output = self.lstm(x)
        # lstm_outs, hidden = self.lstm(x)
        aff_out = self.aff(output) # no ReLU pre-softmax
        score = F.softmax(aff_out, axis=2)

        return score

class GEFF_mxnet(gluon.HybridBlock):
    def __init__(self, output_size=2, hidden_size=600, droprate=0.1, embd_dim=300, **kwargs):
        '''
            Simple baseline model:
            This model use a Fast text word embedding + simple linear layers
        '''
        super(GEFF_mxnet, self).__init__(**kwargs)

        with self.name_scope():
            self.output_size = output_size
            self.hidden_size = hidden_size
            self.embd_dim = embd_dim

            self.fc1 = nn.Dense(self.hidden_size, in_units=self.embd_dim, flatten=False)
            self.fc2 = nn.Dense(1, in_units=self.hidden_size, flatten=False)
            self.fc3 = nn.Dense(1, in_units=self.hidden_size, flatten=False)
            self.fc4 = nn.Dense(self.hidden_size, in_units=self.hidden_size * 2, flatten=False, use_bias=False)
            self.fc5 = nn.Dense(self.hidden_size, in_units=self.hidden_size * 2, flatten=False, use_bias=False)
            self.fc6 = nn.Dense(self.hidden_size, in_units=self.hidden_size, flatten=False)
            self.fc7 = nn.Dense(self.hidden_size, in_units=self.hidden_size, flatten=False)

            self.out = nn.Dense(self.output_size, in_units=self.hidden_size, flatten=False)

    def hybrid_forward(self, F, q1, q2, mask1, mask2):
        '''
         input : qx is a list
                 qx[0] ==> [b, seq_len, num_chars]
                 qx[1] ==> [b, seq_len, dim]
        '''

        # 1. apply FF + RELU
        x1 = F.relu(self.fc1(q1.slice_axis(axis=0, begin=0, end=1)))
        x2 = F.relu(self.fc1(q2))

        # 4. apply dynamic pooling
        x1_hat = F.softmax(self.fc2(x1), axis=1) # [b, seq_len, hidden_size] ==> [b, seq_len, 1]
        x2_hat = F.softmax(self.fc3(x2), axis=1) # [b, seq_len, hidden_size] ==> [b, seq_len, 1]

        # remove padding by masking out padds
        x1_hat = masked_softamx(F, x1_hat, mask1.slice_axis(axis=0, begin=0, end=1), 1)
        x2_hat = masked_softamx(F, x2_hat, mask2, 1)

        x1_hat = F.swapaxes(x1_hat, dim1=1, dim2=2) # ==> [b, seq_len, 1] ==> [b, 1, seq_len]
        x2_hat = F.swapaxes(x2_hat, dim1=1, dim2=2) # ==> [b, seq_len, 1] ==> [b, 1, seq_len]

        u = F.linalg_gemm2(x1_hat, x1) # ==> [b, seq_len,1] * [b, seq_len, hidden_size] ==> [b,1, hidden_size]
        v = F.linalg_gemm2(x2_hat, x2) # ==> [b, seq_len,1] * [b, seq_len, hidden_size] ==> [b,1, hidden_size]

        u = F.squeeze(u, axis=1) # ==> [b,1, hidden_size] ==> [b, hidden_size]
        v = F.squeeze(v, axis=1) # ==> [b,1, hidden_size] ==> [b, hidden_size]

        # 5 . Combine features
        u_mean = F.mean(x1, axis=1) # [b, seq_len, hidden_size]  ==> [b, hidden_size]
        v_mean = F.mean(x2, axis=1) # [b, seq_len, hidden_size]  ==> [b, hidden_size]

        u_combined = F.concat(u, u_mean, dim=1) # [b, hidden_size] ==> [b, 2 * hidden_size]
        v_combined = F.concat(v, v_mean, dim=1) # [b, hidden_size] ==> [b, 2 * hidden_size]

        # 6. Combine question and answer
        x = F.relu(F.broadcast_add(self.fc4(u_combined), self.fc5(v_combined)) + F.broadcast_mul(self.fc6(u), v))
        x = F.relu(self.fc7(x))

        # 7. Get the scores
        out = self.out(x)
        score = F.softmax(out, axis=1)

        return score

def to_numpy(var):
    return (var).detach().cpu().numpy()

def replace_data(mx_param, numpy_data):
    mx_param.set_data(mx.nd.array(numpy_data))

def convert_to_mxnet():
    sys.path.append(os.path.abspath('./src'))
    # from misc.torch_utility import load_model_states

    # Step 1: Load pytroch model
    # print('\nLoading %s' % (model_file))
    # state, params = load_model_states(model_file)

    # Step 2: Initiate a model with the preload weights
    print('\nInit pytorch model ...')
    # import models.GEFF as net
    # pymodel = net.GEFF(hidden_size=params.hidden_size, droprate=params.droprate)
    # pymodel.load_state_dict(state)
    # if use_cuda: pymodel.cuda()

    pymodel = simple_lstm_decoupled.SimpleLSTMDecoupled(
        in_size=300,
        hidden_size=150,
        out_size=2,
        batch_size=1,
        lstm_layers=1,
        use_cuda=False)

    pymodel.load_state_dict(torch.load('./SimpleLSTM_Decoupled', map_location='cpu'))

    print('\nPytorch model:')
    print(pymodel)
    pymodel.summary()

    # Step 4: create the MxNet Model
    print('\nCreate mxnet model ...')
    mxmodel = SimpleLSTM_Decoupled_mxnet(
        in_size=300,
        hidden_size=150,
        out_size=2,
        batch_size=1,
        lstm_layers=1)

    print('\nInitialize mxnet model ...')
    mxmodel.initialize(ctx=mxctx)

    print('\nMxNet model:')
    print(mxmodel)

    # Step 5: copy torch weights to the MxNet
    print('\nConvert model ...')
    replace_data(mxmodel.lstm.l0_h2h_weight, to_numpy(pymodel.lstm.weight_hh_l0))
    replace_data(mxmodel.lstm.l0_h2h_bias, to_numpy(pymodel.lstm.bias_hh_l0))
    replace_data(mxmodel.lstm.l0_i2h_weight, to_numpy(pymodel.lstm.weight_ih_l0))
    replace_data(mxmodel.lstm.l0_i2h_bias, to_numpy(pymodel.lstm.bias_ih_l0))
    replace_data(mxmodel.aff.weight, to_numpy(pymodel.aff.weight))
    replace_data(mxmodel.aff.bias, to_numpy(pymodel.aff.bias))

    total_params = 0
    total_params_Trainable = 0

    for i in pymodel.parameters():
        total_params += np.prod(i.size())
        if (i.requires_grad == True):
            total_params_Trainable += np.prod(i.size())

    print('\nDone:')
    print('  Total number of ALL parameters: %d' % total_params)
    print('  Total number of TRAINABLE parameters: %d' % total_params_Trainable)

    # mxmodel.export('test_mxnet')

    return pymodel, mxmodel

def build_data(B, T, D):
    q1 = np.random.uniform(-1, 1, (B, T, D)).astype(np.float32)
    q2 = q1.copy()

    q1mask = np.ones((B, T), dtype=np.float32)
    q2mask = q1mask.copy()

    for i in range(q1.shape[0]):
        nzeros = random.randint(0, T)
        idx = T-nzeros
        zeros = np.zeros((nzeros, D), dtype=np.float32)
        q1[i,idx:] = zeros
        q2[i,idx:] = zeros
        q1mask[i,idx:] = np.zeros(nzeros, dtype=np.float32)
        q2mask[i,idx:] = np.zeros(nzeros, dtype=np.float32)
        for j in range(idx):
            if random.randint(0, 100) < 20:
                q2[i,j] = np.random.uniform(-1, 1, (1, D)).astype(np.float32)

    return q1, q2, q1mask, q2mask

def Variable(data, *args, **kwargs):
    if use_cuda:
        return autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return autograd.Variable(data, *args, **kwargs)

def profile_pytorch(pymodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy):
    start = datetime.datetime.now()

    q1_len = torch.from_numpy(np.array([q1_numpy.shape[0]]))
    q2_len = torch.from_numpy(np.array([q2_numpy.shape[0]]))
    q1 = Variable(torch.from_numpy(q1_numpy))
    q2 = Variable(torch.from_numpy(q2_numpy))
    q1mask = Variable(torch.from_numpy(q1mask_numpy))
    q2mask = Variable(torch.from_numpy(q2mask_numpy))

    pymodel(q1, q1_len, q2, q2_len, q1mask, q2mask)
    print('.', sep='', end='', flush=True)

    end = datetime.datetime.now()
    return int((end-start).total_seconds()*1000)

def profile_mxnet(mxmodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy):
    start = datetime.datetime.now()

    q1 = mx.nd.array(q1_numpy, dtype='float32')
    q2 = mx.nd.array(q2_numpy, dtype='float32')
    q1mask = mx.nd.array(q1mask_numpy, dtype='float32')
    q2mask = mx.nd.array(q2mask_numpy, dtype='float32')

    scores = mxmodel(q1, q2, q1mask, q2mask)
    scores.wait_to_read()
    print('.', sep='', end='', flush=True)

    end = datetime.datetime.now()
    return int((end-start).total_seconds()*1000)

def profile_mxnet_symbol(mxmodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy):
    data = {'data0':q1_numpy, 'data1':q2_numpy, 'data2':q1mask_numpy, 'data3':q2mask_numpy}
    dataiter = mx.io.NDArrayIter(data, None, 256)

    start = datetime.datetime.now()

    scores = mxmodel.predict(dataiter)
    scores = mx.ndarray.stack(scores).reshape(-1, 2)
    scores.wait_to_read()
    print('.', sep='', end='', flush=True)

    end = datetime.datetime.now()
    return int((end-start).total_seconds()*1000)

def run_mxexport(mxmodel):
    print('\n\nStarting Model Exportation\n=======================================================')

    B = 1
    T = 10
    D = 300

    q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy = build_data(B, T, D)

    # np.save('./exported/q1.npy', q1_numpy)
    # np.save('./exported/q2.npy', q2_numpy)
    # np.save('./exported/q1mask.npy', q1mask_numpy)
    # np.save('./exported/q2mask.npy', q2mask_numpy)

    with mx.Context(mxctx):
        x = mx.nd.array(q1_numpy, dtype='float32')
        q2 = mx.nd.array(q2_numpy, dtype='float32')
        q1mask = mx.nd.array(q1mask_numpy, dtype='float32')
        q2mask = mx.nd.array(q2mask_numpy, dtype='float32')

        mxmodel.init_hidden()

        mxmodel.hybridize()
        mxmodel(x)#, q2, q1mask, q2mask)
        mxmodel.export('./test_mxnet')

    print('\n')

def print_compiler_shapes(shapes):
    print('{', sep='', end='')
    for i, data in enumerate(shapes):
        print("'{}': [".format(data[0]), sep='', end='')
        for j, dim in enumerate(data[1]):
            print('{}'.format(dim), sep='', end='')
            if j != len(data[1])-1:
                print(',', sep='', end='')
        print(']', sep='', end='')
        if i != len(shapes)-1:
            print(', ', sep='', end='')
    print('}', sep='', end='')


def run_mxprofiling(pymodel, mxmodel):
    print('\n\nStarting Model Profiling\n=======================================================')

    B = 3000
    T = 10
    D = 300
    PROFILE_RUNS = 10

    q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy = build_data(B, T, D)

    print('Profiling pymodel in PyTroch ', end='')

    results = []
    for _ in range(PROFILE_RUNS):
        results.append(profile_pytorch(pymodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy))
    print(' computed in {0:,} ms, total of {1:,}'.format(sum(results)//PROFILE_RUNS, sum(results)))

    with mx.Context(mxctx):
        print('Profiling model in MXNet ', end='', sep='')
        results = []
        for _ in range(PROFILE_RUNS):
            results.append(profile_mxnet(mxmodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy))
        print(' computed in {0:,} ms, total of {1:,}'.format(sum(results)//PROFILE_RUNS, sum(results)))

        print('Profiling hybrid model in MXNet ', end='', sep='')
        mxmodel.hybridize()
        results = []
        for _ in range(PROFILE_RUNS):
            results.append(profile_mxnet(mxmodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy))
        print(' computed in {0:,} ms, total of {1:,}'.format(sum(results)//PROFILE_RUNS, sum(results)))

        input_shapes = [('data0', (B,T,D)), ('data1', (B,T,D)), ('data2', (B,T)), ('data3', (B,T))]
        output_shapes = [('out', (B,2))]

        print('MXNet compiler configurations: ')
        print_compiler_shapes(input_shapes)
        print_compiler_shapes(output_shapes)

        print('Profiling symbol model in MXNet ', end='', sep='')
        sym, arg_params, aux_params = mx.model.load_checkpoint('./exported/geff_module', 0)
        mxmodel = mx.mod.Module(symbol=sym, data_names=('data0', 'data1', 'data2', 'data3'), context=mxctx, label_names=None)
        mxmodel.bind(for_training=False, data_shapes=input_shapes, label_shapes=None)
        mxmodel.set_params(arg_params, aux_params, allow_missing=True)


        results = []
        for _ in range(PROFILE_RUNS):
            results.append(profile_mxnet_symbol(mxmodel, q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy))
        print(' computed in {0:,} ms, total of {1:,}'.format(sum(results)//PROFILE_RUNS, sum(results)))

def run_mxverification(pymodel, mxmodel):
    print('\n\nStarting Model Verification\n=======================================================')

    B = 3000
    T = 10
    D = 300

    q1_numpy, q2_numpy, q1mask_numpy, q2mask_numpy = build_data(B, T, D)

    np.save('./exported/q1.npy', q1_numpy)
    np.save('./exported/q2.npy', q2_numpy)
    np.save('./exported/q1mask.npy', q1mask_numpy)
    np.save('./exported/q2mask.npy', q2mask_numpy)

    q1_file = open('./exported/q1.txt', 'w')
    q1_file.write('{0} {1} {2}\n'.format(B, T, D))
    for b in range(B):
        for t in range(T):
            q1_file.write(' '.join(map(str, q1_numpy[b,t])))
            q1_file.write(' ')
        q1_file.write('\n')

    q2_file = open('./exported/q2.txt', 'w')
    q2_file.write('{0} {1} {2}\n'.format(B, T, D))
    for b in range(B):
        for t in range(T):
            q2_file.write(' '.join(map(str, q2_numpy[b,t])))
            q2_file.write(' ')
        q2_file.write('\n')

    q1mask_file = open('./exported/q1mask.txt', 'w')
    q1mask_file.write('{0} {1} {2}\n'.format(B, T, D))
    for b in range(B):
        q1mask_file.write(' '.join(map(str, q1mask_numpy[b])))
        q1mask_file.write('\n')

    q2mask_file = open('./exported/q2mask.txt', 'w')
    q2mask_file.write('{0} {1} {2}\n'.format(B, T, D))
    for b in range(B):
        q2mask_file.write(' '.join(map(str, q2mask_numpy[b])))
        q2mask_file.write('\n')

    q1_len = torch.from_numpy(np.array([q1_numpy.shape[0]]))
    q2_len = torch.from_numpy(np.array([q2_numpy.shape[0]]))
    q1 = Variable(torch.from_numpy(q1_numpy))
    q2 = Variable(torch.from_numpy(q2_numpy))
    q1mask = Variable(torch.from_numpy(q1mask_numpy))
    q2mask = Variable(torch.from_numpy(q2mask_numpy))

    print('Saving run on PyTorch ...')
    scores = pymodel(q1, q1_len, q2, q2_len, q1mask, q2mask)
    with open('./exported/pytorch_run.txt', 'w') as outfile:
        for i in range(q1_numpy.shape[0]):
            outfile.write('q1 [ ')
            for j in range(q1_numpy.shape[1]):
                outfile.write(' '.join(map(str, q1_numpy[i,j])))
                if j < q1_numpy.shape[1] - 1:
                    outfile.write(' ')
            outfile.write(' ]\n')

            outfile.write('q2 [ ')
            for j in range(q2_numpy.shape[1]):
                outfile.write(' '.join(map(str, q2_numpy[i,j])))
                if j < q2_numpy.shape[1] - 1:
                    outfile.write(' ')
            outfile.write(' ]\n')

            outfile.write('q1mask [ ')
            outfile.write(' '.join(map(str, q1mask_numpy[i])))
            outfile.write(' ]\n')

            outfile.write('q2mask [ ')
            outfile.write(' '.join(map(str, q2mask_numpy[i])))
            outfile.write(' ]\n')

            outfile.write('scores [ ')
            outfile.write(' '.join(map(str, scores[i])))
            outfile.write(' ]\n')

    with mx.Context(mxctx):
        q1 = mx.nd.array(q1_numpy, dtype='float32')
        q2 = mx.nd.array(q2_numpy, dtype='float32')
        q1mask = mx.nd.array(q1mask_numpy, dtype='float32')
        q2mask = mx.nd.array(q2mask_numpy, dtype='float32')

        print('Saving run on MXNet ...')
        scores = mxmodel(q1, q2, q1mask, q2mask).asnumpy()
        with open('./exported/mxnet_run.txt', 'w') as outfile:
            for i in range(q1_numpy.shape[0]):
                outfile.write('q1 [ ')
                for j in range(q1_numpy.shape[1]):
                    outfile.write(' '.join(map(str, q1_numpy[i,j])))
                    if j < q1_numpy.shape[1] - 1:
                        outfile.write(' ')
                outfile.write(' ]\n')

                outfile.write('q2 [ ')
                for j in range(q2_numpy.shape[1]):
                    outfile.write(' '.join(map(str, q2_numpy[i,j])))
                    if j < q2_numpy.shape[1] - 1:
                        outfile.write(' ')
                outfile.write(' ]\n')

                outfile.write('q1mask [ ')
                outfile.write(' '.join(map(str, q1mask_numpy[i])))
                outfile.write(' ]\n')

                outfile.write('q2mask [ ')
                outfile.write(' '.join(map(str, q2mask_numpy[i])))
                outfile.write(' ]\n')

                outfile.write('scores [ ')
                outfile.write(' '.join(map(str, scores[i])))
                outfile.write(' ]\n')

if __name__ == "__main__":
    print("\nInitializing ...")
    print("  PyTorch version: {}".format(torch.__version__))
    print("  MXNet version: {}".format(mx.__version__))
    print("  Use Cuda: {}".format(use_cuda))

    pymodel, mxmodel = convert_to_mxnet()
    pymodel.eval()

    if len(sys.argv) > 2:
        print('\nRunning sample test on models:')

        query = sys.argv[2]
        item = sys.argv[3]

        print('  Query:', query)
        print('  Item:', item)

        B = 1
        T = 10
        D = 300

        q1_numpy = np.zeros((B, T, D), dtype=np.float32)
        q2_numpy = np.zeros((B, T, D), dtype=np.float32)
        mask1_numpy = np.ones((B, T), dtype=np.float32)
        mask2_numpy = np.ones((B, T), dtype=np.float32)

        embmodel = ft.load_model('./embs/wiki.en.bin')

        for i, word in enumerate(query.split()):
            emb = embmodel.get_word_vector(word).astype(np.float32)
            q1_numpy[0,i,:] = emb

        for i, word in enumerate(item.split()):
            emb = embmodel.get_word_vector(word).astype(np.float32)
            q2_numpy[0,i,:] = emb

        print('  PyTorch results: ', end='', sep='')
        pyq1 = Variable(torch.from_numpy(q1_numpy))
        pyq2 = Variable(torch.from_numpy(q2_numpy))
        pymask1 = Variable(torch.from_numpy(mask1_numpy))
        pymask2 = Variable(torch.from_numpy(mask2_numpy))

        scores = pymodel(pyq1, None, pyq2, None, pymask1, pymask2)
        print(scores)

        with open('./exported/test_run.txt', 'w') as outfile:
            outfile.write('q1 [ ')
            for j in range(q1_numpy.shape[1]):
                outfile.write(' '.join(map(str, q1_numpy[0,j])))
                if j < q1_numpy.shape[1] - 1:
                    outfile.write(' ')
            outfile.write(' ]\n')

            outfile.write('q2 [ ')
            for j in range(q2_numpy.shape[1]):
                outfile.write(' '.join(map(str, q2_numpy[0,j])))
                if j < q2_numpy.shape[1] - 1:
                    outfile.write(' ')
            outfile.write(' ]\n')

            outfile.write('scores [ ')
            outfile.write(' '.join(map(str, scores[0])))
            outfile.write(' ]\n')

        with mx.Context(mxctx):
            print('  MXNet results: ', end='', sep='')
            mxq1 = mx.nd.array(q1_numpy, dtype='float32')
            mxq2 = mx.nd.array(q2_numpy, dtype='float32')

            mxmask1 = mx.nd.array(mask1_numpy, dtype='float32')
            mxmask2 = mx.nd.array(mask2_numpy, dtype='float32')

            mxscores = mxmodel(mxq1, mxq2, mxmask1, mxmask2).asnumpy()
            print(mxscores)

            print('  MXNet symbol results: ', end='', sep='')
            sym, arg_params, aux_params = mx.model.load_checkpoint('./exported/geff_module', 0)
            mxmodel = mx.mod.Module(symbol=sym, data_names=('data0', 'data1', 'data2', 'data3'), context=mxctx, label_names=None)
            mxmodel.bind(for_training=False, data_shapes=[('data0', (B,T,D)), ('data1', (B,T,D)), ('data2', (B,T)), ('data3', (B,T))], label_shapes=None)
            mxmodel.set_params(arg_params, aux_params, allow_missing=True)

            data = {'data0':q1_numpy, 'data1':q2_numpy, 'data2':mask1_numpy, 'data3':mask2_numpy}
            dataiter = mx.io.NDArrayIter(data, None, 1)

            symscores = mxmodel.predict(dataiter)
            symscores = mx.ndarray.stack(symscores).reshape(-1, 2)
            symscores.wait_to_read()

            print(symscores.asnumpy())

    else:
        run_mxexport(mxmodel)
        # run_mxverification(pymodel, mxmodel)
        # run_mxprofiling(pymodel, mxmodel)