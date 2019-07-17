import dataset_groups
import data_util
import dd_platform


d1_name = 'ASI.201809-201812'
d2_name = 'ASI.201809-201812.v107'

d1_utt_2_conf, d1_utt_2_truth = data_util.load_posteriors_file(dataset_groups.groups[d1_name]['results']['stage1'])
d2_utt_2_conf, d2_utt_2_truth = data_util.load_posteriors_file(dataset_groups.groups[d2_name]['results']['stage1'])

d1_keys = set(d1_utt_2_conf.keys())
d2_keys = set(d2_utt_2_conf.keys())

pruned = d2_keys - d1_keys

for uttid in pruned:
    print uttid
    d1_utt_2_conf[uttid] = 0
    d1_utt_2_truth[uttid] = d2_utt_2_truth[uttid]

filename = "%s/%s_EmptyUttsReplaced_s1_results.csv" % (dd_platform.DATA_LOC, d2_name)
data_util.write_posteriors_file(filename, d1_utt_2_conf, d1_utt_2_truth)



