import numpy as np


def get_metric_statistics(values, repeat_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(repeat_times)
    return mean, conf_interval

def transform(M, tc_std ,tc_mean, tc_mean_cls, tc_std_cls):
    raw_m = M * tc_std + tc_mean
    return (raw_m - tc_mean_cls) / tc_std_cls