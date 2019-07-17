import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import data_util
import numpy as np
import sklearn.metrics as metrics
import dd_platform
import lr_scheds.lr_scheds as lr_scheds

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt


def train_epoch(optimizer, criterion, train_loader, model):
    model.train()
    
    # smooth_avg_loss = 0
    batch_losses = []
    
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target[:, 0])

        # smooth_avg_loss = smooth_avg_loss * 0.9 + loss.data[0].item() * 0.1
        batch_losses.append(loss.data[0].item())
        
        # backprop
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
        #     print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
        #         epoch, batch_idx + 1, ave_loss)
        #     if (batch_idx + 1) == len(train_loader):
        #         train_losses.append(ave_loss)
    
    avg_loss = np.mean(batch_losses)
    
    print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
        epoch, batch_idx + 1, avg_loss)
    
    return batch_losses, avg_loss#, smooth_avg_loss


def run_eval(dataset_loader, model, name='test'):
    model.eval()

    batch_losses = []
    TN, FP, FN, TP = 0, 0, 0, 0

    for batch_idx, (x, target) in enumerate(dataset_loader):
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
        
        # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
        #     print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
        #         epoch, batch_idx + 1, ave_loss)
        #     if (batch_idx + 1) == len(train_loader):
        #         train_losses.append(ave_loss)
    
    avg_loss = np.mean(batch_losses)
    acc = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    precision = 1.0 * TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = 1.0 * TP / (TP + FN)
    fpr = 1.0 * FP / (FP + TN)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print '==>>> epoch: {}, {} loss: {:.6f}, acc: {:.3f})'.format(epoch, name, avg_loss, acc)
    
    return batch_losses, avg_loss, acc, precision, recall, fpr, f1


# DATA
data_loc = dd_platform.DATA_LOC
word_2_embedding = data_util.load_embedding_as_dict('s3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec')
# test_loader = data_util.get_torch_dataset_loader('%s/prodv1/test.index' % data_loc, word_2_embedding)
# dev_loader = data_util.get_torch_dataset_loader('%s/prodv1/dev.index' % data_loc, word_2_embedding)
# train_loader = data_util.get_torch_dataset_loader('%s/prodv1/train.index' % data_loc, word_2_embedding)
# test_loader = data_util.get_torch_dataset_loader('%s/mixed/test.prodv1_ADS.Week43-44.index' % data_loc, word_2_embedding)
# dev_loader = data_util.get_torch_dataset_loader('%s/mixed/dev.prodv1_ADS.Week41-42.index' % data_loc, word_2_embedding)
# train_loader = data_util.get_torch_dataset_loader('%s/mixed/train.prodv1_ADS.Week34-40.index' % data_loc, word_2_embedding, batch_size=BATCH_SIZE)
# print '==>>> total training batch number: {}'.format(len(train_loader))
# print '==>>> total testing batch number: {}'.format(len(test_loader))


from models.baseline_bow_mlp import BOWMLP
from configs.bow_mlp_v1 import *

dataset_group = 'ASI.201809-201811'
# dataset_group = 'Prod.v104'
dataset_type = 'test'

import dataset_groups
datasets = dataset_groups.groups[dataset_group][dataset_type]
newDatasetName = '%s.%s' % (dataset_group, dataset_type)

test_loader = data_util.get_torch_dataset_loader(dataset_groups.groups[dataset_group]['test'], word_2_embedding)
dev_loader = data_util.get_torch_dataset_loader(dataset_groups.groups[dataset_group]['dev'], word_2_embedding)
train_loader = data_util.get_torch_dataset_loader(dataset_groups.groups[dataset_group]['train'], word_2_embedding, batch_size=BATCH_SIZE)
print '==>>> total training batch number: {}'.format(len(train_loader))
print '==>>> total testing batch number: {}'.format(len(test_loader))
print '==>>> total dev batch number: {}'.format(len(dev_loader))



# TORCH
use_cuda = torch.cuda.is_available()
print 'Using Cuda?', use_cuda
torch.manual_seed(SEED)


# MODEL

model = BOWMLP(300, 2, 600)
if use_cuda:
    model = model.cuda()


# OPTIMIZER/LOSS
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
class_weights = torch.FloatTensor(CLASS_WEIGHTS)
if use_cuda:
    class_weights = class_weights.cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)


# TRAINING
train_losses = []
dev_losses = []
dev_accs = []
test_losses = []
test_accs = []
test_fprs = []
test_recalls = []
test_f1s = []
lr_decays = 0
plateaus = []


# TRAINING
train_losses = []
dev_losses = []
dev_accs = []

# lr_scheduler = lr_scheds.XValPlateauLRScheduler(optimizer, LR, LR_PLATEAUS_HISTORY, LR_PLATEAUS_TO_DECAY, LR_MAX_DECAYS, LR_DECAY_FACTOR)
lr_scheduler = lr_scheds.LRTestScheduler(optimizer, min=0.0, max=0.2, step=0.01)

for epoch in xrange(MAX_EPOCHS):
    # training
    batch_losses, avg_loss = train_epoch(optimizer, criterion, train_loader, model)
    train_losses.append(avg_loss)
    batch_losses, avg_loss, acc, _precision, _recall, _fpr, _f1 = run_eval(dev_loader, model, name='dev')
    dev_losses.append(avg_loss)
    dev_accs.append(acc)

    # udpate LR
    lr_scheduler.update_learning_rate(dev_losses)
    if lr_scheduler.is_done():
        break


with open('results.csv', 'w') as outfile:
    outfile.write('train_losses {} \n'.format(' '.join([str(x) for x in train_losses])))
    outfile.write('dev_losses {} \n'.format(' '.join([str(x) for x in dev_losses])))
    outfile.write('dev_accs {} \n'.format(' '.join([str(x) for x in dev_accs])))

# for epoch in xrange(MAX_EPOCHS):
#     # training
#     batch_losses, avg_loss = train_epoch(optimizer, criterion, train_loader, model)
#     train_losses.append(avg_loss)
#     batch_losses, avg_loss, acc, precision, recall, fpr, f1 = run_eval(dev_loader, model, name='dev')
#     dev_losses.append(avg_loss)
#     dev_accs.append(acc)
#     batch_losses, avg_loss, acc, precision, recall, fpr, f1 = run_eval(test_loader, model)
#     test_losses.append(avg_loss)
#     test_accs.append(acc)
#     test_fprs.append(fpr)
#     test_recalls.append(recall)
#     test_f1s.append(f1)
#
#     # decay LR if necessary
#     if len(dev_losses) >= 2:
#         plateaus.append(1 if (dev_losses[-1] > dev_losses[-2]) else 0)
#
#     if len(plateaus) >= LR_PLATEAUS_HISTORY and sum(plateaus[-LR_PLATEAUS_HISTORY:]) >= LR_PLATEAUS_TO_DECAY:
#         del plateaus[:]
#         if lr_decays == LR_MAX_DECAYS:
#             break
#         lr_decays += 1
#         new_lr = LR * LR_DECAY_FACTOR
#         print 'decaying learning rate {} ==> {}'.format(LR, new_lr)
#         LR = new_lr
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = LR


# with open('train-results.csv', 'w') as outfile:
#     outfile.write('train_losses {} \n'.format(' '.join([str(x) for x in train_losses])))
#     outfile.write('dev_losses {} \n'.format(' '.join([str(x) for x in dev_losses])))
#     outfile.write('dev_accs {} \n'.format(' '.join([str(x) for x in dev_accs])))
#     outfile.write('test_losses {} \n'.format(' '.join([str(x) for x in test_losses])))
#     outfile.write('test_accs {} \n'.format(' '.join([str(x) for x in test_accs])))
#     outfile.write('test_fprs {} \n'.format(' '.join([str(x) for x in test_fprs])))
#     outfile.write('test_recalls {} \n'.format(' '.join([str(x) for x in test_recalls])))
#     outfile.write('test_f1s {} \n'.format(' '.join([str(x) for x in test_f1s])))
#
#
# plt.plot(train_losses, label='train loss')
# plt.plot(test_losses, label='test loss')
# plt.plot(test_accs, label='test accs')
# plt.legend()
# plt.show()

torch.save(model.state_dict(), model.name())
