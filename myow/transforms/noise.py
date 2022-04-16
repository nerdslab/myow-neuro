import torch


class Noise:
    r"""Adds Gaussian noise to neural activity. The firing rate vector needs to have already been normalized, and
        the Gaussian noise is center and has standard deviation of :obj:`std`.

    Args:
        std (float): Standard deviation of Gaussian noise.
    """
    def __init__(self, std, apply_p=1., same_on_trial=False, same_on_batch=False):
        self.std = std
        self.apply_p = apply_p

        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                noise = torch.normal(0.0, self.std, size=(x.size(1),), device=x.device)
                x = x + noise
        elif self.same_on_trial:
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    noise = torch.normal(0.0, self.std, size=(x.size(1),), device=x.device)
                    x[trial_mask] += noise
        else:
            noise = torch.normal(0.0, self.std, size=x.size(), device=x.device)
            # cancel noise based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            noise[apply_mask] = 0.
            x = x + noise
        return x

    def __repr__(self):
        return '{}(std={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.std,
                                                                                   self.apply_p, self.same_on_trial,
                                                                                   self.same_on_batch)


class Pepper:
    r"""Adds a constant to the neuron firing rate with a probability of :obj:`p`.

    Args:
        p (float, Optional): Probability of adding pepper. (default: :obj:`0.5`)
        apply_p (float, Optional): Probability of applying the transformation. (default: :obj:`1.0`)
        std (float, Optional): Constant to be added to neural activity. (default: :obj:`1.0`)
    """
    def __init__(self, p=0.5, c=1.0, apply_p=1., same_on_trial=True, same_on_batch=False):
        self.p = p
        self.c = c
        self.apply_p = apply_p

        self.same_on_trial = same_on_trial
        self.same_on_batch = same_on_batch

    def __call__(self, x, trial=None):
        if self.same_on_batch or (trial is None and self.same_on_trial):
            if torch.rand(1) < self.apply_p:
                pepper_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                x = x + self.c * pepper_mask
        elif self.same_on_trial:
            for trial_id in torch.unique(trial):
                if torch.rand(1) < self.apply_p:
                    trial_mask = trial == trial_id
                    pepper_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
                    x[trial_mask] += self.c * pepper_mask
        else:
            pepper_mask = torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1) < self.p
            # cancel pepper based on apply probability
            apply_mask = torch.rand(x.size(0)) < 1 - self.apply_p
            pepper_mask[apply_mask] = False
            x = x + self.c * pepper_mask
        return x

    def __repr__(self):
        return '{}(p={}, c={}, apply_p={}, same_on_trial={}, same_on_batch={})'.format(self.__class__.__name__, self.p,
                                                                                       self.c, self.apply_p,
                                                                                       self.same_on_trial,
                                                                                       self.same_on_batch)
