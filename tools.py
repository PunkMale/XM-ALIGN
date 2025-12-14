'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch, warnings, glob, yaml, logging
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F
import torch.nn as nn


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    return dict(yaml_config, **kwargs)


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
    
    def write(self, message):
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
            f.flush()
    
    def print(self, message):
        print(message)
        self.write(message)


def init_system(args):
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')
    args.submission_save_path = os.path.join(args.save_path, 'submission')
    args.model_save_path_a    = os.path.join(args.save_path, 'model_a')
    args.model_save_path_v    = os.path.join(args.save_path, 'model_v')
    os.makedirs(args.submission_save_path, exist_ok = True)
    os.makedirs(args.model_save_path_a, exist_ok = True)
    os.makedirs(args.model_save_path_v, exist_ok = True)

    args.modelfiles_a = glob.glob('%s/model_0*.model'%args.model_save_path_a)
    args.modelfiles_v = glob.glob('%s/model_0*.model'%args.model_save_path_v)
    args.modelfiles_a.sort()
    args.modelfiles_v.sort()
    args.score_file = open(args.score_save_path, "a+")
    args.logger = Logger(args.log_path)
    return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
