import torch


class Dropout:
    r"""Drops a neuron with a probability of :obj:`p`. Inplace!

    Args:
        p (float, Optional): Probability of dropout. (default: :obj:`0.5`)
        apply_p (float, Optional): Probability of applying the transformation. (default: :obj:`1.0`)
    """
    def __init__(self, p: float = 0.5, apply_p=1., same_on_trial=False, same_on_batch=False):
        self.p = p
        self.apply_p = apply_p

        assert (not same_on_batch) or (same_on_batch and same_on_trial)
        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                dropout_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x[:, dropout_mask] = 0

        elif self.same_on_trial:
            dropout_mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    dropout_mask[trial_mask] = \
                        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            x[dropout_mask] = 0
        else:
            dropout_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            dropout_mask[apply_mask] = False
            x[dropout_mask] = 0
        return x

    def __repr__(self):
        return '{}(p={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                 self.apply_p, self.same_on_trial,
                                                                                 self.same_on_batch)


class RandomizedDropout:
    def __init__(self, p: float = 0.5, apply_p=1., same_on_trial=False, same_on_batch=False):
        self.p = p
        self.apply_p = apply_p

        assert (not same_on_batch) or (same_on_batch and same_on_trial)
        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                p = torch.rand(1) * self.p
                dropout_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
                x[:, dropout_mask] = 0

        elif self.same_on_trial:
            dropout_mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    p = torch.rand(1) * self.p
                    dropout_mask[trial_mask] = \
                        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
            x[dropout_mask] = 0
        else:
            # generate a random dropout probability for each sample
            p = torch.rand(x.size(0)) * self.p
            # generate dropout mask
            dropout_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < p.view((-1, 1))
            # cancel dropout based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            dropout_mask[apply_mask] = False
            x[dropout_mask] = 0
        return x

    def __repr__(self):
        return '{}(p={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                 self.apply_p, self.same_on_trial,
                                                                                 self.same_on_batch)
