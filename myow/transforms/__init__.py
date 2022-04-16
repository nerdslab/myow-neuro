from .compose import Compose
from .normalize import Normalize
from .dropout import Dropout, RandomizedDropout
from .noise import Noise, Pepper


def get_neural_transform(*, normalize=None, dropout=None, randomized_dropout=None, noise=None, pepper=None, noise_after_norm=False):
    assert not(dropout is not None and randomized_dropout is not None)

    transforms = []
    if dropout is not None and dropout['p'] != 0. and dropout['apply_p'] != 0.:
        transforms.append(Dropout(**dropout))
    if randomized_dropout is not None and randomized_dropout['p'] != 0. and randomized_dropout['apply_p'] != 0:
        transforms.append(RandomizedDropout(**randomized_dropout))
    if pepper is not None and pepper['p'] != 0 and pepper['c'] != 0 and pepper['apply_p'] != 0:
        transforms.append(Pepper(**pepper))
    if noise is not None and noise['std'] !=0 and noise['apply_p'] !=0:
        transforms.append(Noise(**noise))
    if not noise_after_norm:
        transforms.append(Normalize(**normalize))
    else:
        transforms.insert(-1, Normalize(**normalize))

    transform = Compose(transforms)
    return transform

import inspect
from collections import namedtuple


def cfg(cls):
    args = inspect.signature(cls).parameters
    fields, defaults = [], []
    for name, param in args.items():
        fields.append(name)
        defaults.append(param.default)
    return namedtuple(cls.__name__+'Cfg', fields, defaults=defaults)

"""
Normalize.cfg = cfg(Normalize)
Dropout.cfg = cfg(Dropout)
RandomizedDropout.cfg = cfg(RandomizedDropout)
Noise.cfg = cfg(Noise)
Pepper.cfg = cfg(Pepper)
"""
