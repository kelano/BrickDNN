"""
Script for performing domain analysis (NLU Intents) on a given dataset
"""

import data_util
import numpy as np
import dataset_groups
import dd_platform
import re

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt


# Mapping to align common intents renamed over time
INTENT_MAPPING = {
    '^PhaticQAIntent$': 'PhaticIntent',
    'QAIntent': 'QAIntent'
}


def get_intent_to_class_counts(utt_2_intent, minimum_filter=None):
    intent_2_pos = {}
    intent_2_neg = {}
    intent_2_tot = {}

    for uttid, intent in utt_2_intent.items():
        for search_pattern, replacement in INTENT_MAPPING.items():
            if re.search(search_pattern, intent) is not None:
                intent = replacement
                break

        if intent not in intent_2_tot:
            intent_2_pos[intent] = 0
            intent_2_neg[intent] = 0
            intent_2_tot[intent] = 0

        truth = utt_2_truth[uttid]
        if truth == 0:
            intent_2_neg[intent] += 1
        else:
            intent_2_pos[intent] += 1
        intent_2_tot[intent] += 1

    # filter by minimum if specified
    if minimum_filter is not None:
        for intent in list(intent_2_tot.keys()):
            if intent_2_tot[intent] < minimum_filter:
                del intent_2_pos[intent]
                del intent_2_neg[intent]
                del intent_2_tot[intent]

    return intent_2_pos, intent_2_neg, intent_2_tot



# dataset_group = 'Prod.v104'
dataset_group = 'ASI.201809-201812'
# dataset_group = 'Prod.v100'
# dataset_group = 'WBR.Local'
dataset_name = 'test'

dataset = dataset_groups.groups[dataset_group]
utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(dataset[dataset_name])

intent_2_pos, intent_2_neg, intent_2_tot = get_intent_to_class_counts(utt_2_intent, minimum_filter=50)

print '%d Total Intents' % len(intent_2_tot.keys())

sorted_pos_counts = [i[1] for i in sorted(intent_2_pos.items(), key=lambda (k, v): k)]
sorted_neg_counts = [i[1] for i in sorted(intent_2_neg.items(), key=lambda (k, v): k)]
sorted_tot_counts = [i[1] for i in sorted(intent_2_tot.items(), key=lambda (k, v): k)]

sorted_keys = [i[0] for i in sorted(intent_2_tot.items(), key=lambda (k, v): k)]

ind = np.arange(len(intent_2_tot.keys()))  # the x locations for the groups
width = 0.35       # the width of the bars

# plt.bar(ind, intent_2_neg.values(), width, label='NDD')
# plt.bar(ind + width, intent_2_pos.values(), width, label='DD')

sorted_neg_percentage_counts = np.array(sorted_neg_counts, dtype=np.float32) / np.sum(sorted_neg_counts)
sorted_pos_percentage_counts = np.array(sorted_pos_counts, dtype=np.float32) / np.sum(sorted_pos_counts)
sorted_tot_percentage_counts = np.array(sorted_tot_counts, dtype=np.float32) / np.sum(sorted_tot_counts)

for tup in list(reversed(sorted(zip(sorted_tot_percentage_counts, sorted_keys), key=lambda tup: tup[0]))):
    print tup[1],',',tup[0]

# plt.bar(ind, sorted_tot_percentage_counts, width, label=dataset_name)
plt.bar(ind, sorted_pos_percentage_counts, width, label='DD')
plt.bar(ind + width, sorted_neg_percentage_counts, width, label='NDD')


# plt.xticks(ind + width / 2, intent_2_neg.keys(), rotation=90)
plt.xticks(ind + width / 2, sorted_keys, rotation=90)
# plt.set_xticklabels(intent_2_neg.keys())

# plt.hist((pos_intents, neg_intents), label=('DD', 'NDD'))
# plt.xticks(rotation=90)
plt.title('Domain Analysis %s.%s' % (dataset_group, dataset_name))
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('DomainAnalysis-%s-%s.png' % (dataset_group, dataset_name))
