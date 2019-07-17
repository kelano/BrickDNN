"""
Data analysis tools (PCA, Confidence Histograms, Individual Token Analysis)
"""

import data_util
import numpy as np
import dd_platform
import dataset_groups
import trained_model_groups
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import os

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt



def analyze_utterances(utt_2_sentence, utt_2_truth):
    """
    For analyzing specific token usages within a dataset. Use helper lambdas to modify search type:
        is_only_maker: for searching for utterances consisting only of the token of interest
        initated_maker: for utterances starting with token of interest
        not_initiated_maker: utterances containing token anywhere other than start
        anywhere_maker: utterances containing token anywhere
    :param utt_2_sentence:
    :param utt_2_truth:
    """
    total = len(utt_2_truth.values())
    process = lambda filter: [utt_2_truth[x] for x in utt_2_sentence.keys() if filter(utt_2_sentence[x])]

    all = lambda: lambda y: True
    is_only_maker = lambda x: lambda y: x == y
    is_only_maker.__name__ = 'is_only'
    initiated_maker = lambda x: lambda y: len(y.split()) > 1 and y.split()[0] == x
    initiated_maker.__name__ = 'initiated'
    not_initiated_maker = lambda x: lambda y: len(y.split()) > 1 and y.split()[0] != x and y.split().count(x) >= 1
    not_initiated_maker.__name__ = "not-initiated"
    anywhere_maker = lambda x: lambda y: y.split().count(x) >= 1
    anywhere_maker.__name__ = "anywhere"

    res = process(all())
    print '%15s:\tDD: %8d\tNDD: %8d\tn: %8d (%.3f%%)' % \
          ('ALL', res.count('False'), res.count('True'), len(res), float(len(res)) / total * 100)

    words = ['alexa', 'stop', 'okay', 'yes', 'no', '', 'call']
    makers = [is_only_maker, initiated_maker, not_initiated_maker, anywhere_maker]
    for word in words:
        print word
        for maker in makers:
            res = process(maker(word))
            print '\t%15s:\tDD: %8d\tNDD: %8d\tn: %8d (%.3f%%)' % \
                  (maker.__name__, res.count(1), res.count(0), len(res), float(len(res)) / total * 100)


def generate_histogram(utt_2_data, utt_2_truth=None, data_lambda=None, normed=True, bins=20, title='Histogram'):
    """
    Generate a posterior histogram for a given set of posteriors
    :param utt_2_data: per-utterance data
    :param utt_2_truth: per-utterance truth (1 for DD 0 for NDD), if not provided, combines classes
    :param data_lambda: lambda to do any value modification
    """
    if utt_2_truth is not None:
        pos = []
        neg = []
        for uttid in utt_2_data.keys():
            data = data_lambda(utt_2_data[uttid]) if data_lambda is not None else utt_2_data[uttid]

            # this is to handle empty utterance cases where np mean of token confs returns nan
            if np.isnan(data):
                continue

            truth = utt_2_truth[uttid]
            if truth == 1:
                pos.append(data)
            else:
                neg.append(data)

        plt.hist(pos, bins=bins, color='g', alpha=0.3, normed=normed, label='DD')
        plt.hist(neg, bins=bins, color='r', alpha=0.3, normed=normed, label='NDD')
        plt.title('%s per Class' % title)
    else:
        all = []
        for uttid in utt_2_data.keys():
            data = data_lambda(utt_2_data[uttid]) if data_lambda is not None else utt_2_data[uttid]

            # this is to handle empty utterance cases where np mean of token confs returns nan
            if np.isnan(data):
                continue

            all.append(data)

        plt.hist(all, bins=bins, color='b', alpha=0.3, label='ALL')
        plt.title('%s' % title)
    plt.legend()
    plt.show()
    plt.savefig('%s/%s-histogram%s.png' % (results_loc, title, '-perClass' if utt_2_truth is not None else ''))


def get_pos_neg_common_words(utt_2_sentence, utt_2_truth):
    all_pos = []
    all_neg = []
    for utt in utt_2_sentence.keys():
        sentence = utt_2_sentence[utt]
        truth = utt_2_truth[utt] == 1
        if truth:
            for word in sentence.split():
                all_pos.append(word)
        else:
            for word in sentence.split():
                all_neg.append(word)

    all_pos_set = set(all_pos)
    all_neg_set = set(all_neg)

    common = all_pos_set.intersection(all_neg_set)
    just_pos = all_pos_set.difference(common)
    just_neg = all_neg_set.difference(common)

    print 'Pos: %d Neg: %d Common: %d' % (len(just_pos), len(just_neg), len(common))
    return just_pos, just_neg, common


def is_valid(confidences):
    if len(confidences) == 0:
        return False
    for confidence in confidences:
        if np.isnan(confidence):
            return False
    return True


def visualize_pca(pos_words, neg_words, embedding, dim=2, labels=False):
    x = np.array(embedding.values())
    targets = []
    for key in embedding.keys():  # [0:10]:
        if key in pos_words:
            targets.append('pos')
        elif key in neg_words:
            targets.append('neg')
        else:
            targets.append('common')

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=dim)
    principalComponents = pca.fit_transform(x)
    columns = ['PC%d' % d for d in range(1, dim + 1)]
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=columns)

    fig = plt.figure(figsize=(8, 8))
    if dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = Axes3D(fig)
    else:
        ax = fig.add_subplot(1,1,1)
    
    ax.set_xlabel(columns[0], fontsize=15)
    ax.set_ylabel(columns[1], fontsize=15)
    if dim == 3:
        ax.set_zlabel(columns[2], fontsize=15)
    ax.set_title('%d Component PCA' % dim, fontsize=20)

    finalDf = pd.concat([principalDf, pd.DataFrame(data=targets, columns=['target'])['target']], axis=1)
    # targets = ['neg', 'pos']  # , 'common']
    targets = ['pos', 'neg']
    # colors = ['r', 'g']  # , 'b']
    colors = ['g', 'r']  # , 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(*[finalDf.loc[indicesToKeep, column] for column in columns], c=color, s=10)

    if labels == True:
        for i in range(0, len(finalDf['PC1'])):
            if finalDf['target'][i] == 'common':
                continue
            if np.random.random_sample() < 0.9:
                continue
            ax.text(*[finalDf.loc[i, column] for column in columns], s=embedding.keys()[i], zorder=1)

    ax.legend(targets)
    ax.grid()
    plt.show()
    fig.savefig('%s/PCA%dD' % (results_loc, dim))


results_loc = os.getcwd() + '/results'
if not os.path.exists(results_loc):
    os.makedirs(results_loc)

model_group = 'Prod.v104'
model_name = 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw'

# dataset_group = 'Prod.v104'
dataset_group = 'ASI.201809-201812'
# dataset_group = 'Prod.v100'
# dataset_group = 'WBR.Local'
dataset_name = 'test'

dataset = dataset_groups.groups[dataset_group]

utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(dataset[dataset_name])
s1_utt_2_conf, s1_utt_2_truth = data_util.load_posteriors_file(dataset['results']['stage1'])

# posterior analysis
generate_histogram(s1_utt_2_conf, s1_utt_2_truth, title='Posterior Histogram')
generate_histogram(s1_utt_2_conf, title='Posterior Histogram')

# utterance length analysis
generate_histogram(utt_2_confidences, utt_2_truth, lambda confidences: len(confidences), title='Utterance Length Histogram')
generate_histogram(utt_2_confidences, data_lambda=lambda confidences: len(confidences), title='Utterance Length Histogram')

# mean token confidence analysis
generate_histogram(utt_2_confidences, utt_2_truth, lambda confidences: np.mean(confidences), title='Avg Token ASR Confidence Histogram')
generate_histogram(utt_2_confidences, data_lambda=lambda confidences: np.mean(confidences), title='Avg Token ASR Confidence Histogram')

# token analysis
analyze_utterances(utt_2_sentence, utt_2_truth)

# PCA
embedding = data_util.load_embedding_as_dict(trained_model_groups.models[model_group][model_name]['embedding'])
pos_words, neg_words, common_words = get_pos_neg_common_words(utt_2_sentence, utt_2_truth)
visualize_pca(pos_words, neg_words, embedding, dim=2, labels=True)
visualize_pca(pos_words, neg_words, embedding, dim=3, labels=True)
