import torch


class Normalize:
    r"""Normalization transform. Also removes dead neurons

    Args:
        mean (torch.Tensor): Mean.
        std (torch.Tensor): Standard deviation.
    """
    def __init__(self, mean, std):
        if isinstance(mean, float):
            mean = torch.tensor([mean])  # prevent 0 sized tensors

        if isinstance(std, float):
            std = torch.tensor([std])  # prevent 0 sized tensors

        self.not_dead_mask = std != 0
        self.mean = mean[self.not_dead_mask]
        self.std = std[self.not_dead_mask]

    def __call__(self, x, trial=None):
        return (x[:, self.not_dead_mask] - self.mean.to(x)) / self.std.to(x)

    def __repr__(self):
        return '{}(dim={})'.format(self.__class__.__name__, self.mean.size())
