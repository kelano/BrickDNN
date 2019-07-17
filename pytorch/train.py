import pytorch.data_util as data_util
import torch
import fasttext
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics
import numpy as np



def train_epoch(optimizer, criterion, train_loader, model, reset_hidden=True):
    model.train()

    batch_losses = []
    batch_update = len(train_loader) / 10

    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if reset_hidden:
            model.hidden = model.init_hidden()

        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target.squeeze())
        batch_losses.append(loss.data.item())

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # if batch_idx % batch_update == 0:
        #     print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, np.mean(batch_losses)))

    avg_loss = np.mean(batch_losses)

    print('==>>> epoch: {}, train loss: {:.6f}'.format(epoch, avg_loss))

    return batch_losses, avg_loss


def run_eval(dataset_loader, model, reset_hidden=True, name='test'):
    model.eval()

    batch_losses = []
    batch_update = len(dataset_loader) / 10
    TN, FP, FN, TP = 0, 0, 0, 0
    correct = 0
    total = 0

    for batch_idx, (x, target) in enumerate(dataset_loader):
        if reset_hidden:
            model.hidden = model.init_hidden()

        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target.squeeze())
        _, pred_label = torch.max(out.data, 1)

        # tn, fp, fn, tp = metrics.confusion_matrix(target[:, 0], pred_label).ravel()
        # TN += tn
        # FP += fp
        # FN += fn
        # TP += tp

        total += len(pred_label)
        correct += len(np.where(target.squeeze() == pred_label)[0])

        batch_losses.append(loss.data.item())

        # if batch_idx % batch_update == 0:
        #     print('==>>> epoch: {}, batch index: {}, {} loss: {:.6f}'.format(epoch, batch_idx, name, np.mean(batch_losses)))

    avg_loss = np.mean(batch_losses)
    acc = 1.0 * correct / total

    print('==>>> epoch: {}, {} loss: {:.6f}, acc: {:.3f})'.format(epoch, name, avg_loss, acc))

    return batch_losses, avg_loss, acc


mystr = str('hello THER3')
print(mystr)
mystr = mystr.lower()
print(mystr)


print('loading fasttest')
word_2_vec, word_to_idx, idx_to_vec, embedding_mat = fasttext.load_vectors()

# print(word_to_idx['UNK'])
# print(word_2_vec['UNK'])
# print(word_2_vec['unk'])

print('Embedding len: ', len(word_to_idx))

# exit()

use_cuda = False if sys.platform == 'darwin' else torch.cuda.current_device()
print('CUDA', use_cuda)


df = data_util.load_data()

X = df.Name
Y = df.Theme


# print(type(X))
# X = data_util.to_idx_rep(X, word_to_idx)
# X = data_util.to_vec_rep(X, word_2_vec)

X2 = data_util.to_elmo_vec_rep(X, word_2_vec)

print(X2.shape, X2[0].shape)
# print(type(X))

Y, one_hot_array = data_util.to_one_hot(Y)

nClasses = len(one_hot_array)

X = X.to_numpy()
Y = Y.to_numpy()

print(X.shape)
print(Y.shape)

# print(X.to_numpy().shape)
# print(Y.to_numpy().shape)

# exit()

# print(X)
# print(Y)

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_util.partition_data(X, Y)

print('Train: ', len(X_train))
print('Dev: ', len(X_dev))
print('Test: ', len(X_test))

train_loader = data_util.get_torch_dataset_loader(X_train, Y_train, batch_size=64)
dev_loader = data_util.get_torch_dataset_loader(X_dev, Y_dev, batch_size=64)
test_loader = data_util.get_torch_dataset_loader(X_test, Y_test, batch_size=64)

# X_dev_sub = X_dev[0:10]

# X_new = data_util.to_idx_rep(X_dev_sub, word_to_idx)

# print(X_new)

import pytorch.simple_lstm as simple_lstm
model = simple_lstm.SimpleLSTM(
            embedding_dim=300,
            hidden_size=150,
            out_size=nClasses,
            batch_size=64,
            lstm_layers=3,
            use_cuda=use_cuda)



# OPTIMIZER/LOSS/LR SCHEDULER
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.CrossEntropyLoss()

import pytorch.lr_scheds as lr_scheds
lr_scheduler = lr_scheds.XValPlateauLRScheduler(optimizer, 0.05, 3, 2, 2, 0.1)
# lr_scheduler = lr_scheds.LRTestScheduler(optimizer, lr_min=0.0, lr_max=0.2, lr_step=0.01)


# TRAINING
train_losses = []
dev_losses = []
dev_accs = []
for epoch in range(100):
    # training
    batch_losses, avg_loss = train_epoch(optimizer, criterion, train_loader, model)
    train_losses.append(avg_loss)
    batch_losses, avg_loss, acc = run_eval(dev_loader, model, name='dev')
    dev_losses.append(avg_loss)
    dev_accs.append(acc)

    # udpate LR
    lr_scheduler.update_learning_rate(dev_losses)
    if lr_scheduler.is_done():
        break

run_eval(test_loader, model, name='test')
