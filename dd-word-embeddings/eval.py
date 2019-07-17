import data_util
import dd_platform
import two_stage_model_util
import dataset_groups
import trained_model_groups
import os
import stats_util
import generate_posteriors

import matplotlib
matplotlib.use(dd_platform.MATPLOTLIB_USE)
import matplotlib.pyplot as plt



def evaluate_model_per_intent(utt_2_conf, utt_2_truth, utt_2_intent, min_required=500):
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
    evaluate_models(nptuples, utt_2_truth)


def collect_stats(name, posteriors, targets, threshold_fns):
    print name
    eval_stats = stats_util.EvalStats(name, posteriors, targets)
    eval_stats.collect_thresholds(threshold_fns)
    return eval_stats


def evaluate_models(np_tuples, utt_2_truth, ops, threshold_fns):
    results_loc = os.getcwd() + '/results'
    if not os.path.exists(results_loc):
        os.makedirs(results_loc)

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
            t.append(1 if utt_2_truth[k] else 0)

        model_2_stats[name] = collect_stats(name, p, t, threshold_fns)

    # write results
    first = True
    with open('%s/%s' % (results_loc, 'results.csv'), 'w') as outfile:
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
            plt.scatter(ts_v.far, 1 - ts_v.frr,  s=50, label='%s_%s' % (name, ts_k))
            if 'ThresholdOP' in ts_k and 'stage1' in name:
                plt.axvline(ts_v.far)
                plt.axhline(1 - ts_v.frr)
        plt.plot(stats.fpr, stats.tpr, label=name)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (%s)' % ', '.join(model_2_stats.keys()))
    plt.legend()
    plt.show()
    plt.savefig('%s/%s' % (results_loc, 'ROC'))


def main():
    """
    Default evaluation configuration
    """

    # specify individual model group/name pairs here
    model_name_tups = [
        ['Prod.v104', 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw'],
        # ['Prod.v107', 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw'],
    ]

    # ...or run all models from a given group
    # model_group = 'Prod.v104'
    # for model_name in trained_model_groups.models[model_group].keys():
    #     model_name_tups.append((model_group, model_name))

    # specify dataset group/name
    # dataset_group = 'Prod.v104'
    # dataset_group = 'Prod.v107'
    dataset_group = 'ASI.201809-201812.v107'
    # dataset_group = 'Prod.v100'
    # dataset_group = 'WBR.Week34-40'
    dataset_name = 'test'

    dataset = dataset_groups.groups[dataset_group]

    np_tuples = []

    # load stage one results and add to model name/posteriors tuples
    s1_utt_2_conf, utt_2_truth = data_util.load_posteriors_file(dataset['results']['stage1'])
    np_tuples.append(('stage1', s1_utt_2_conf))
    # print 'stage1', len(s1_utt_2_conf)

    # specify stage one operating points for analysis and two-stage evaluation
    # import configs.stage1_prodv1 as s1
    # import configs.stage1_prodv104 as s1
    # import configs.stage1_prodv107 as s1
    import configs.stage1_prodv107_fixed as s1
    ops = [
        s1.THRESHOLD,
        # .750,
        # .250,
        # .350,
        # .210,
        # .105,
    ]
    # ops = np.linspace(0.22, 0.42, 11)

    # specify threshold functions for FA/FR operating points to test on all models
    threshold_fns = [
        stats_util.MaxBACC(), # maximize balanced accuracy
        # stats_util.FRRThreshold(.213),
        # stats_util.FARThreshold(.064),
        # stats_util.FARThreshold(.1),
        # stats_util.FARThreshold(.1),
        stats_util.FARThreshold(s1.FAR * .9, 'FARThreshold_10%RelRed'),
        stats_util.FARThreshold(s1.FAR * .8, 'FARThreshold_20%RelRed'),
    ]
    # add stage one OPs to threshold functions
    for op in ops:
        threshold_fns.append(stats_util.ThresholdOP(op))

    for model_name_tup in model_name_tups:
        model_group, model_name = model_name_tup[0], model_name_tup[1]

        # add NLU DD model running independently of stage one
        s2_utt_2_conf, s2_utt_2_truth = generate_posteriors.generate_posteriors(model_group, model_name, dataset_group, dataset_name)
        np_tuples.append(('.'.join(model_name_tup), s2_utt_2_conf))

        # add NLU DD model running in two-stage fashion with stage one at each OP
        print 'generating two stage utt 2 conf'
        for op in ops:
            two_stage_utt_2_conf = two_stage_model_util.gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth, op)
            np_tuples.append(('two_stage_%f_%s' % (op, '.'.join(model_name_tup)), two_stage_utt_2_conf))

    # generate metrics
    print 'generating metrics'
    evaluate_models(np_tuples, utt_2_truth, ops, threshold_fns)
    # evaluate_model_per_intent(s2_utt_2_conf, utt_2_truth, utt_2_intent)


if __name__ == "__main__":
    main()
