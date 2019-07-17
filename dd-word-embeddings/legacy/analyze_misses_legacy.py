import dd_platform
import data_util
import two_stage_model_util
from subprocess import call
import dataset_groups



model_group = 'Prod.v100'
# model_name = 'SimpleLSTM_150H_05005'
model_name = 'BOW-MLP'
#dataset_group = 'Prod.v104'
dataset_group = 'ASI.201809-201811'
# dataset_group = 'Prod.v100'

dataset_name = 'test'

analyze_s1 = False
analyze_two_stage = False


# model_path = 'BOW-MLP'
# # model_path = 'BOW-MLP-MixedTrain'
# # model_path = 'SimpleLSTM_150H'
# # model_path = 'SimpleLSTM_150H_TL'
# model_name = model_path.split('/')[-1]
#
data_loc = dd_platform.DATA_LOC
#
data_type = 'prodv1'
# data_type = 'ADS'
# # data_type = 'mixed'
#
dataset_name = 'test'
# dataset_name = 'test.ADS.Week43-44'
# dataset_name = 'test.prodv1_ADS.Week43-44'
# dataset_name = 'test.ADS.with_WW.with_Empty.Week43-44'




#
# import dataset_groups
# import generate_posteriors
#
#
# dataset = dataset_groups.groups[dataset_group]
# # dataset = dataset_groups.groups["v104"]
#
# utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(dataset[dataset_name])
# s1_utt_2_conf, s1_utt_2_truth = data_util.load_hover_dory_results(dataset['results']['stage1'])
# s2_utt_2_conf, s2_utt_2_truth = generate_posteriors.generate_posteriors(model_group, model_name, dataset_group,
#                                                                         dataset_name)
#
threshold = .500
#
# if analyze_s1:
#     utt_2_conf = s1_utt_2_conf
#     utt_2_truth = s1_utt_2_truth
# elif analyze_two_stage:
#     int_utt_2_truth = data_util.intersection(s1_utt_2_truth, s2_utt_2_truth)
#     utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, s1_utt_2_truth)
#     utt_2_truth = int_utt_2_truth
# else:
#     utt_2_conf = s2_utt_2_conf
#     utt_2_truth = s2_utt_2_truth



index = '%s/%s/%s.index' % (data_loc, data_type, dataset_name)
results_pre = '%s/%s/%s.%s-results.csv' % (data_loc, data_type, dataset_name, '%s')

word_2_embedding = data_util.load_embedding_as_dict('%s/embeddings/wiki-news-300d-1M-subset.vec' % data_loc)

# test_loader = data_util.get_torch_dataset_loader(index, word_2_embedding)

s1_utt_2_conf, s1_utt_2_truth = data_util.load_dory_results(results_pre % 'stage1')
s2_utt_2_conf, s2_utt_2_truth = data_util.load_dory_results(results_pre % model_name)

print len(s1_utt_2_conf), len(s1_utt_2_truth)
print len(s2_utt_2_conf), len(s2_utt_2_truth)

# int_utt_2_truth = data_util.intersection(s1_utt_2_truth, s2_utt_2_truth)
s1_utt_2_conf, s2_utt_2_conf = data_util.get_intersection_items(s1_utt_2_conf, s2_utt_2_conf)

print len(s1_utt_2_conf), len(s1_utt_2_truth)
print len(s2_utt_2_conf), len(s2_utt_2_truth)

# two_stage_utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, s1_utt_2_truth)

utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_dataset(index)

ALL_MISSES = {}
for k, v in s2_utt_2_conf.items():
    # if v == 0:
    #     continue
    # if k not in utt_2_truth:
    #     continue

    # dd_truth = s1_utt_2_truth[k] == 1
    dd_truth = utt_2_truth[k] == 'False'
    # print utt_2_truth[k], type(utt_2_truth[k]), dd_truth
    # exit()
    #
    # if dd_truth != dd_truth_2:
    #     print 'PROBLEM'
    #     exit()

    dd_pred = v >= threshold
    if dd_truth != dd_pred and dd_truth is False:
        if utt_2_sentence[k] == 'thank you':
            print v, dd_pred, dd_truth
        # print v, 'truth:%s' % ('DD' if dd_truth else 'NDD'), utt_2_sentence[k]
        # continue

        dk = "%s\t%s\t%s" % (utt_2_sentence[k], utt_2_intent[k], 'truth:%s' % ('DD' if dd_truth is True else 'NDD'))
        # if utt_2_sentence[k] == '':
        #     print k
            # call('grep %s /Users/kelleng/data/dd/dd-data/*' % k, shell=True)
            # continue

        if dk not in ALL_MISSES:
            ALL_MISSES[dk] = 1
        else:
            ALL_MISSES[dk] = ALL_MISSES[dk] + 1

sd = sorted(ALL_MISSES.items(), key=lambda i: i[1])
for miss in sd[-20:]:
    print miss[0], miss[1]