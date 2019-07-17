import data_util
import numpy as np
import sklearn.metrics as metrics
import dd_platform
import two_stage_model_util

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt




def evaluate_per_intent(utt_2_conf, utt_2_truth, utt_2_intent, min_required=500):
    intent_2_utt_2_conf = {}
    for k, v in utt_2_conf.items():
        intent = utt_2_intent[k]
        if intent not in intent_2_utt_2_conf:
            intent_2_utt_2_conf[intent] = {}
        intent_2_utt_2_conf[intent][k] = v
    nptuples = []
    for intent, intent_utt_2_conf in intent_2_utt_2_conf.items():
        if len(intent_utt_2_conf) > min_required:
            nptuples.append(('%s-%d' % (intent, len(intent_utt_2_conf)), intent_utt_2_conf))
    gen_metrics_multi(nptuples, utt_2_truth)


def collect_stats(name, posteriors, targets, threshold_fns):
    eval_stats = stats_util.EvalStats(name, posteriors, targets)
    eval_stats.collect_thresholds(threshold_fns)
    return eval_stats


def gen_metrics_multi_new(np_tuples, utt_2_truth, ops, threshold_fns):
    # collect stats
    model_2_stats = {}
    # model_2_threshold_stats = {}
    for np_tuple in np_tuples:
        name = np_tuple[0]
        posteriors = np_tuple[1]
        # print name

        k_p = set(posteriors.keys())
        k_t = set(utt_2_truth.keys())
        k_common = set.intersection(k_p, k_t)

        p = []
        t = []
        for k in k_common:
            p.append(posteriors[k])
            t.append(1 if utt_2_truth[k] == 'False' else 0)
        model_2_stats[name] = collect_stats(name, p, t, threshold_fns)

        # model_2_threshold_stats[name] = {}
        # for threshold_fn in thresholds:
        #     threshold_stats = threshold_fn(model_2_stats[name])
        #     model_2_threshold_stats[name][threshold_stats.name()] = threshold_stats

        # if name == 'stage1':
        #     for op in ops:
        #         op_name = '%s OP %f' % (name, op)
        #         model_2_stats[op_name] = collect_stats(op_name, posteriors, targets, op)

    # write results
    first = True
    with open('results.csv', 'w') as outfile:
        for name, stats in model_2_stats.items():
            if first:
                outfile.write('%s,%s' % (stats.header(), stats.threshold_stats[stats.threshold_stats.keys()[0]].header()))
                outfile.write('\n')
                first = False
            for ts_k, ts_v in stats.threshold_stats.items():
                outfile.write('%s,%s' % (str(stats), str(ts_v)))
                outfile.write('\n')

    # ROC
    for name, stats in model_2_stats.items():
        for ts_k, ts_v in stats.threshold_stats.items():
            if 'ThresholdOP' in ts_k:
                continue
            plt.scatter(ts_v.far, 1 - ts_v.frr,  s=50, label='%s_%s' % (name, ts_k))
        plt.plot(stats.fpr, stats.tpr, label=name)

    for name, stats in model_2_stats.items():
        for ts_k, ts_v in stats.threshold_stats.items():
            if 'ThresholdOP' not in ts_k:
                continue
            if 'ThresholdOP' in ts_k and 'Stage1' not in name:
                continue
            plt.scatter(ts_v.far, 1 - ts_v.frr,  s=50, label='%s_%s' % (name, ts_k))
            if 'ThresholdOP' in ts_k and 'Stage1' in name:
                plt.axvline(ts_v.far)
                plt.axhline(1 - ts_v.frr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (%s)' % ', '.join(model_2_stats.keys()))
    plt.legend()
    plt.show()
    plt.savefig('ROC')




def gen_metrics_multi(np_tuples, utt_2_truth, ops):
    with open('results.csv', 'w') as outfile:
        outfile.write('model', 'max_bACC', 'FAR', 'FRR', 'AUC')
        for np_tuple in np_tuples:
            name = np_tuple[0]
            print name
            posteriors = np_tuple[1]
            targets = [(0 if utt_2_truth[uttid] == 'True' else 1) for uttid in posteriors.keys()]

            fpr, tpr, thresholds = metrics.roc_curve(targets, posteriors.values(), 1)
            precision, recall, pr_thresholds = metrics.precision_recall_curve(targets, posteriors.values(), 1)

            # trim unusable points
            precision = precision[1:]
            recall = recall[1:]
            # pr_thresholds = pr_thresholds[1:]
            if 'two_stage' in name:
                fpr = fpr[:-1]
                tpr = tpr[:-1]
                thresholds = thresholds[:-1]

            tnr = 1 - fpr
            fnr = 1 - tpr
            bAcc = (tpr + tnr) / 2
            max_bAcc_index = np.argmax(bAcc)
            max_bAcc = bAcc[max_bAcc_index]
            max_bAcc_threshold = thresholds[max_bAcc_index]
            # f1 = 2 * precision * recall / (precision + recall)

            print '\tMax bACC: %.3f Threshold: %.3f FAR: %.3f FRR: %.3f' % \
                  (max_bAcc, max_bAcc_threshold, fpr[max_bAcc_index], fnr[max_bAcc_index])
            if 'two_stage' in name:
                plt.scatter(fpr[max_bAcc_index], tpr[max_bAcc_index], s=50, marker='^', label='%s-Max_bAcc' % name)

            plt.plot(fpr, tpr, label=name)



            # p, r, f, s = metrics.precision_recall_fscore_support(targets, posteriors.values())
            # print name, 'max f1', pr_thresholds[np.argmax(f1)], f1[np.argmax(f1)]

            # closest_to_max_bAcc = np.argmin(np.abs(thresholds - pr_thresholds[np.argmax(f1)]))
            #     plt.scatter(fpr[closest_to_maxfpr], tpr[closest_to_maxfpr], s=50, marker='^', label='%s-MaxF1' % name)
            # print '%s Max bACC FAR' % name, fpr[closest_to_maxfpr], ' FRR', 1 - tpr[closest_to_maxfpr]

            # tn, fp, fn, tp = metrics.confusion_matrix(targets, posteriors > ).ravel()

            # thresholds[0] = 1
            # pr_thresholds[0] = 1

            auc = metrics.auc(fpr, tpr)
            print '\tAUC', auc
            # thresholds[0] = 1.0

            outfile.write()

        for op in ops:
            print('S1 OP %f' % op)
            # get stage 1
            for np_tuple in np_tuples:
                if np_tuple[0] == 'stage1':
                    posteriors = np_tuple[1]
                    targets = [(0 if utt_2_truth[uttid] == 'True' else 1) for uttid in posteriors.keys()]

                    fpr, tpr, thresholds = metrics.roc_curve(targets, posteriors.values(), 1)

                    tnr = 1 - fpr
                    fnr = 1 - tpr
                    bAcc = (tpr + tnr) / 2

                    closest_to_op_index = np.argmin(np.abs(thresholds - op))
                    bAcc = bAcc[closest_to_op_index]
                    bAcc_threshold = thresholds[closest_to_op_index]
                    # f1 = 2 * precision * recall / (precision + recall)

                    print '\tbACC: %.3f Threshold: %.3f FAR: %.3f FRR: %.3f' % \
                          (bAcc, bAcc_threshold, fpr[closest_to_op_index], fnr[closest_to_op_index])

                    # if 'two stage' in [t[0] for t in np_tuples]:
                    plt.scatter(fpr[closest_to_op_index], tpr[closest_to_op_index], s=50, label='S1 OP %f' % op)
                    plt.axvline(fpr[closest_to_op_index])
                    plt.axhline(tpr[closest_to_op_index])


    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (%s)' % ', '.join([np_tuple[0] for np_tuple in np_tuples]))
    plt.legend()
    plt.show()
    plt.savefig('ROC')
    
    # plt.plot(pr_thresholds, f1)
    # plt.show()

    for np_tuple in np_tuples:
        name = np_tuple[0]
        posteriors = np_tuple[1]
        targets = [(0 if utt_2_truth[uttid] == 'True' else 1) for uttid in posteriors.keys()]

        fpr, tpr, thresholds = metrics.roc_curve(targets, posteriors.values(), 1)
        # p, r, f, s = metrics.precision_recall_fscore_support(all_target.data[:, 0], all_pred_labels)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(targets, posteriors.values(), 1)
        thresholds[0] = 1
        pr_thresholds[0] = 1

        aucpr = metrics.auc(recall, precision)
        print name, 'aucpr', aucpr
        thresholds[0] = 1.0
        plt.plot(recall, precision, label=name)
    plt.legend()
    plt.show()
    plt.savefig('PR')

    for np_tuple in np_tuples:
        name = np_tuple[0]
        posteriors = np_tuple[1]
        targets = [(0 if utt_2_truth[uttid] == 'True' else 1) for uttid in posteriors.keys()]

        fpr, tpr, thresholds = metrics.roc_curve(targets, posteriors.values(), 1)
        # p, r, f, s = metrics.precision_recall_fscore_support(all_target.data[:, 0], all_pred_labels)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(targets, posteriors.values(), 1)
        
        thresholds[0] = 1
        pr_thresholds[0] = 1

        eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
        print name, 'eer', eer
        plt.plot(thresholds, fpr, label="FPR-%s" % name)
        plt.plot(thresholds, 1 - tpr, label="FNR-%s" % name)
    plt.legend()
    plt.show()
    plt.savefig('EER')


model_names = [
    # 'BOW-MLP',
    # 'BOW-MLP-MixedTrain',
    # 'SimpleLSTM_TEST',
    # 'SimpleLSTM_150H',
    # 'SimpleLSTM_150H_1PAT',
    # 'SimpleLSTM_150H_2Stack',
    # 'SimpleLSTM_150H_05005',
    'SimpleLSTM_150H_TL',
    # 'SimpleLSTM_150H_TL_FT',
]

rename_dict = {
    'SimpleLSTM_150H_05005': 'SimpleLSTM',
    'SimpleLSTM_150H_TL': 'SimpleLSTM_TL'
}

data_loc = dd_platform.DATA_LOC

# data_type = 'prodv1'
# dataset_name = 'test'
#
data_type = 'ADS'
dataset_name = 'test.ADS.with_WW.with_Empty.Week43-44'
# dataset_name = 'test.ADS.with_WW.Week43-44'
# dataset_name = 'test.ADS.Week43-44'

# data_type = 'mixed'
# dataset_name = 'test.prodv1_ADS.Week43-44'

index = '%s/%s/%s.index' % (data_loc, data_type, dataset_name)
results_pre = '%s/%s/%s.%s-results.csv' % (data_loc, data_type, dataset_name, '%s')
utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_dataset(index)

np_tuples = []

print results_pre % 'stage1'
s1_utt_2_conf, s1_utt_2_truth = data_util.load_dory_results(results_pre % 'stage1')
np_tuples.append(('Stage1', s1_utt_2_conf))
print 'stage1', len(s1_utt_2_conf)

import configs.stage1_prodv1 as s1
ops = [
    s1.THRESHOLD,
    # .5,
    # .375,
    # .350,
    # .325,
    # .300,
    # .275,
    # .250,
    # .225,
    # .200,
    # .175,
    # .150,
    # .125,
    .100,
    # .210,
    # .105,
]
# ops = range(.2, .42, .01)
# ops = np.linspace(0.22, 0.42, 11)

import stats_util
threshold_fns = [
    # stats_util.MaxBACC(),
    stats_util.FRRThreshold(s1.FRR),
    stats_util.FARThreshold(s1.FAR),
    # stats_util.FARThreshold(s1.FAR * .95, 'FARThreshold_5%RelRed'),
    # stats_util.FARThreshold(s1.FAR * .9, 'FARThreshold_10%RelRed'),
    # stats_util.FARThreshold(s1.FAR * .85, 'FARThreshold_15%RelRed'),
    # stats_util.FARThreshold(s1.FAR * .8, 'FARThreshold_20%RelRed'),
]
for op in ops:
    threshold_fns.append(stats_util.ThresholdOP(op))
    # two_stage_utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s1_utt_2_conf, utt_2_truth, op)
    # np_tuples.append(('two_stage_%f_%s' % (op, 'Stage1'), two_stage_utt_2_conf))

for model_name in model_names:
    # print model_name, len(s2_utt_2_conf)

    # s2
    s2_utt_2_conf, s2_utt_2_truth = data_util.load_dory_results(results_pre % model_name)

    if model_name in rename_dict:
        model_name = rename_dict[model_name]
    # np_tuples.append((model_name, s2_utt_2_conf))

    # two stage
    print 'generating two stage utt 2 conf'
    for op in ops:
        two_stage_utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth, op)
        np_tuples.append(('two_stage_%f_%s' % (op, model_name), two_stage_utt_2_conf))

    # misc
    # two_stage_utt_2_conf = two_stage_model_util.gen_mean_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth)
    # np_tuples.append(('mean_two_stage_%s' % model_name, two_stage_utt_2_conf))
    # two_stage_utt_2_conf = two_stage_model_util.gen_harmonic_mean_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth)
    # np_tuples.append(('harmonic_mean_two_stage_%s' % model_name, two_stage_utt_2_conf))
    # two_stage_utt_2_conf = two_stage_model_util.gen_or_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth)
    # np_tuples.append(('max_two_stage_%s' % model_name, two_stage_utt_2_conf))
    # two_stage_utt_2_conf = two_stage_model_util.gen_min_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth)
    # np_tuples.append(('min_two_stage_%s' % model_name, two_stage_utt_2_conf))
    # two_stage_utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth, .315)
    # np_tuples.append(('two_stage_HALFOP_%s' % model_name, two_stage_utt_2_conf))

print 'generating metrics'
gen_metrics_multi_new(np_tuples, utt_2_truth, ops, threshold_fns)
# evaluate_per_intent(s2_utt_2_conf, utt_2_truth, utt_2_intent)
