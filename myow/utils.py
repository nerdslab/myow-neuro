import logging
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

def seed_everything(seed):
    if seed is not None:
        logging.info("Global seed set to {}.".format(seed))

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def collect_params(modules, exclude_bias_and_bn=True):
    """Collects all parameters from :obj:`modules`.

    .. note::
        In the PyTorch implementation of ResNet, `downsample.1` are bn layers, hence they are excluded
        if :obj:`exclude_bias_and_bn` is set to :obj:`True`.
    """
    param_list = []
    for module in modules:
        for name, param in module.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list
