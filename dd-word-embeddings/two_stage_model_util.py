"""
Utility for generating two stage posteriors given stage 1 (ASR) and stage 2 (NLU) model scores
"""

def gen_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth, s1_threshold):
    '''
    Default AND implementation, if s1 score > threshold then use s2 score, otherwise use 0
    :param s1_utt_2_conf:
    :param s2_utt_2_conf:
    :param utt_2_truth:
    :param s1_threshold:
    :return:
    '''
    s3_utt_2_conf = {}
    for key in utt_2_truth.keys():
        if key not in s1_utt_2_conf or key not in s2_utt_2_conf:
            continue
        s3_utt_2_conf[key] = 0 if s1_utt_2_conf[key] < s1_threshold else s2_utt_2_conf[key]
    return s3_utt_2_conf


### ALTERNATIVE SCORE COMBINATION APPROACHES


def gen_mean_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth):
    s3_utt_2_conf = {}
    for key in utt_2_truth.keys():
        if key not in s1_utt_2_conf or key not in s2_utt_2_conf:
            continue
        s3_utt_2_conf[key] = (s1_utt_2_conf[key] + s2_utt_2_conf[key]) / 2
    return s3_utt_2_conf


def gen_harmonic_mean_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth):
    s3_utt_2_conf = {}
    for key in utt_2_truth.keys():
        if key not in s1_utt_2_conf or key not in s2_utt_2_conf:
            continue
        s3_utt_2_conf[key] = (2 * s1_utt_2_conf[key] * s2_utt_2_conf[key]) / (s1_utt_2_conf[key] + s2_utt_2_conf[key])
    return s3_utt_2_conf


def gen_or_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth):
    s3_utt_2_conf = {}
    for key in utt_2_truth.keys():
        if key not in s1_utt_2_conf or key not in s2_utt_2_conf:
            continue
        s3_utt_2_conf[key] = max(s1_utt_2_conf[key], s2_utt_2_conf[key])
    return s3_utt_2_conf


def gen_min_two_stage_utt_2_conf(s1_utt_2_conf, s2_utt_2_conf, utt_2_truth):
    s3_utt_2_conf = {}
    for key in utt_2_truth.keys():
        if key not in s1_utt_2_conf or key not in s2_utt_2_conf:
            continue
        s3_utt_2_conf[key] = min(s1_utt_2_conf[key], s2_utt_2_conf[key])
    return s3_utt_2_conf

