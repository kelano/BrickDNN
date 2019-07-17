import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sklearn.metrics as metrics
import os

import data_util
import dd_platform
import lr_scheds.lr_scheds as lr_scheds
import dataset_groups
import trained_model_groups
import pytorch_util

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt


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
        loss = criterion(out, target[:, 0])
        batch_losses.append(loss.data[0].item())
        
        # backprop
        loss.backward()
        
        # update weights
        optimizer.step()

        if batch_idx % batch_update == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, np.mean(batch_losses))
    
    avg_loss = np.mean(batch_losses)
    
    print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
        epoch, batch_idx + 1, avg_loss)
    
    return batch_losses, avg_loss


def run_eval(dataset_loader, model, reset_hidden=True, name='test'):
    model.eval()

    batch_losses = []
    batch_update = len(train_loader) / 10
    TN, FP, FN, TP = 0, 0, 0, 0

    for batch_idx, (x, target) in enumerate(dataset_loader):
        if reset_hidden:
            model.hidden = model.init_hidden()

        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target[:, 0])
        _, pred_label = torch.max(out.data, 1)
        
        tn, fp, fn, tp = metrics.confusion_matrix(target[:, 0], pred_label).ravel()
        TN += tn
        FP += fp
        FN += fn
        TP += tp
        
        batch_losses.append(loss.data[0].item())

        if batch_idx % batch_update == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, np.mean(batch_losses))
    
    avg_loss = np.mean(batch_losses)
    acc = 1.0 * (TP + TN) / (TP + TN + FP + FN)

    print '==>>> epoch: {}, {} loss: {:.6f}, acc: {:.3f})'.format(epoch, name, avg_loss, acc)
    
    return batch_losses, avg_loss, acc


# MAIN

model_group = 'Prod.v107'

# model_name = 'SimpleLSTM_150H_05005'
# model_name = 'SimpleLSTM_150H_2Stack_TL'
model_name = 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw'

dataset_group = 'ASI.201809-201812'
# dataset_group = 'Prod.v104'
# dataset_group = 'Prod.v107'
# dataset_group = '"Mixed_ASI.201809-201812_Prod.v104"'


# embeddings
embedding = trained_model_groups.models[model_group][model_name]['embedding']
# embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec'
embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat(embedding)
word_2_embedding = data_util.load_embedding_as_dict(embedding)


# TORCH
use_cuda = torch.cuda.is_available()
print 'Using Cuda?', use_cuda


# MODEL
model, conf = pytorch_util.create_model_instance(model_group, model_name, embedding_mat, use_cuda)
if use_cuda:
    model = model.cuda()


# CONF
SEED = conf['SEED']
BATCH_SIZE = conf['BATCH_SIZE']
LR = conf['LR']
MOMENTUM = conf['MOMENTUM']
CLASS_WEIGHTS = conf['CLASS_WEIGHTS']
LR_PLATEAUS_HISTORY = conf['LR_PLATEAUS_HISTORY']
LR_PLATEAUS_TO_DECAY = conf['LR_PLATEAUS_TO_DECAY']
LR_MAX_DECAYS = conf['LR_MAX_DECAYS']
LR_DECAY_FACTOR = conf['LR_DECAY_FACTOR']
MAX_EPOCHS = conf['MAX_EPOCHS']
MODEL_NAME = conf['MODEL_NAME']

torch.manual_seed(SEED)

# DATA
dev_loader = data_util.get_torch_sequence_dataset_loader(dataset_groups.groups[dataset_group]['dev'], word_2_idx, batch_size=BATCH_SIZE)
train_loader = data_util.get_torch_sequence_dataset_loader(dataset_groups.groups[dataset_group]['train'], word_2_idx, batch_size=BATCH_SIZE)
print '==>>> total training batch number: {}'.format(len(train_loader))
# print '==>>> total testing batch number: {}'.format(len(test_loader))
print '==>>> total dev batch number: {}'.format(len(dev_loader))


# OPTIMIZER/LOSS/LR SCHEDULER
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
class_weights = torch.FloatTensor(CLASS_WEIGHTS)
if use_cuda:
    class_weights = class_weights.cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

lr_scheduler = lr_scheds.XValPlateauLRScheduler(optimizer, LR, LR_PLATEAUS_HISTORY, LR_PLATEAUS_TO_DECAY, LR_MAX_DECAYS, LR_DECAY_FACTOR)
# lr_scheduler = lr_scheds.LRTestScheduler(optimizer, lr_min=0.0, lr_max=0.2, lr_step=0.01)


# TRAINING
train_losses = []
dev_losses = []
dev_accs = []
for epoch in xrange(MAX_EPOCHS):
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

# make results dir
results_loc = os.getcwd() + '/results'
if not os.path.exists(results_loc):
    os.makedirs(results_loc)

# save training results
with open('%s/training-results-%s.csv' % (results_loc, model_name), 'w') as outfile:
    outfile.write('train_losses {} \n'.format(' '.join([str(x) for x in train_losses])))
    outfile.write('dev_losses {} \n'.format(' '.join([str(x) for x in dev_losses])))
    outfile.write('dev_accs {} \n'.format(' '.join([str(x) for x in dev_accs])))

# plot training results
plt.plot(train_losses, label='train loss')
plt.plot(dev_losses, label='dev loss')
plt.plot(dev_accs, label='dev accs')
plt.legend()
plt.savefig('%s/training-results-fig-%s' % (results_loc, model_name))

# save model
torch.save(model.state_dict(), MODEL_NAME)
