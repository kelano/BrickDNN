"""
Tool for analyzing incorrect classifications of a model on a specified dataset
"""


import data_util
import two_stage_model_util
import dataset_groups
import generate_posteriors


model_group = 'Prod.v104'
model_name = 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw'

# dataset_group = 'Prod.v104'
dataset_group = 'ASI.201809-201812'
# dataset_group = 'Prod.v100'
# dataset_group = 'WBR'
dataset_name = 'test'

# Defaults to evaluating an NLU DD model standalone, use these variables to test ASR DD or two stage instead.
analyze_s1 = False
analyze_two_stage = False

# show N most frequent misses
n = 20

dataset = dataset_groups.groups[dataset_group]

utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(dataset[dataset_name])
s1_utt_2_conf, s1_utt_2_truth = data_util.load_posteriors_file(dataset['results']['stage1'])
s2_utt_2_conf, s2_utt_2_truth = generate_posteriors.generate_posteriors(model_group, model_name, dataset_group,
                                                                        dataset_name)

threshold = .500

if analyze_s1:
    utt_2_conf = s1_utt_2_conf
    utt_2_truth = s1_utt_2_truth
elif analyze_two_stage:
    utt_2_truth = data_util.intersection(s1_utt_2_truth, s2_utt_2_truth)
    utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth, s1_threshold=0.5)
else:
    utt_2_conf = s2_utt_2_conf
    utt_2_truth = s2_utt_2_truth

print len(utt_2_truth), len(utt_2_conf)

ALL_MISSES = {}
for k, v in utt_2_conf.items():
    dd_truth = utt_2_truth[k] == 1

    dd_pred = v >= threshold
    if dd_truth != dd_pred:# and dd_truth is False:
        dk = "%s\t%s" % (utt_2_sentence[k], 'truth:%s' % ('DD' if dd_truth is True else 'NDD'))
        # if utt_2_sentence[k] == '':
        #     print k
            # call('grep %s /Users/kelleng/data/dd/dd-data/*' % k, shell=True)
            # continue

        if dk not in ALL_MISSES:
            ALL_MISSES[dk] = 1
        else:
            ALL_MISSES[dk] = ALL_MISSES[dk] + 1

sd = sorted(ALL_MISSES.items(), key=lambda i: i[1])
for miss in sd[-n:]:
    print miss[0], miss[1]