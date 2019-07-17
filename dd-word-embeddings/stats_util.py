import numpy as np
import sklearn.metrics as metrics


class ThresholdStats:
    def __init__(self, name, threshold, bAcc, far, frr):
        self.name = name
        self.threshold = threshold
        self.bAcc = bAcc
        self.far = far
        self.frr = frr

    def __str__(self):
        return ','.join([str(var) for var in (self.name, self.threshold, self.bAcc, self.far, self.frr)])

    def header(self):
        return 'threshold_name,threshold,bAcc,far,frr'


class ThresholdFn:
    def __init__(self, name):
        self.name = name

    def get_index(self, stats):
        return -1

    def __call__(self, stats):
        index = self.get_index(stats)
        return ThresholdStats(
            self.name,
            stats.thresholds[index],
            stats.bAcc[index],
            stats.fpr[index],
            stats.fnr[index]
        )


class MaxBACC(ThresholdFn):
    def __init__(self):
        ThresholdFn.__init__(self, 'MaxBACC')

    def get_index(self, stats):
        return np.argmax(stats.bAcc)


class ThresholdOP(ThresholdFn):
    def __init__(self, op):
        ThresholdFn.__init__(self, 'ThresholdOP_%f' % op)
        self.op = op

    def get_index(self, stats):
        return np.argmin(np.abs(self.op - stats.thresholds))


class FRRThreshold(ThresholdFn):
    def __init__(self, op):
        ThresholdFn.__init__(self, 'FRRThreshold_%f' % op)
        self.op = op

    def get_index(self, stats):
        return np.argmin(np.abs(self.op - stats.fnr))


class FARThreshold(ThresholdFn):
    def __init__(self, op, name='FARThreshold'):
        ThresholdFn.__init__(self, '%s_%f' % (name, op))
        self.op = op

    def get_index(self, stats):
        return np.argmin(np.abs(self.op - stats.fpr))



class EvalStats:

    def __init__(self, name, posteriors, targets):
        self.name = name

        self.p = targets.count(1)
        self.n = targets.count(0)

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(targets, posteriors, 1)
        self.precision, self.recall, self.pr_thresholds = metrics.precision_recall_curve(targets, posteriors, 1)

        # trim unusable points
        self.precision = self.precision[1:]
        self.recall = self.recall[1:]
        # pr_thresholds = pr_thresholds[1:]
        if 'two_stage' in name:
            self.fpr = self.fpr[:-1]
            self.tpr = self.tpr[:-1]
            self.thresholds = self.thresholds[:-1]

        self.tnr = 1 - self.fpr
        self.fnr = 1 - self.tpr
        self.bAcc = (self.tpr + self.tnr) / 2

        self.auc = metrics.auc(self.fpr, self.tpr)
        self.aucpr = metrics.auc(self.recall, self.precision)
        self.eer = self.fpr[np.argmin(np.abs(self.fpr - self.fnr))]

    def collect_thresholds(self, threshold_fns):
        self.threshold_stats = {}
        for threshold_fn in threshold_fns:
            self.threshold_stats[threshold_fn.name] = threshold_fn(self)

    def __str__(self):
        return ','.join([str(var) for var in (self.name, self.p, self.n, self.auc, self.aucpr, self.eer)])

    def header(self):
        return 'name,pos,neg,auc,aucpr,eer'







