import torch

def compute_representations(encoder, x, device):
    x = x.to(device)
    with torch.no_grad():
        repr = encoder(x)
    return repr


def batch_iter(X, *tensors, batch_size=256):
    r"""Creates iterator over tensors.

    Args:
        X (torch.tensor): Feature tensor (shape: num_instances x num_features).
        tensors (torch.tensor): Target tensors (shape: num_instances).
        batch_size (int, Optional): Batch size. (default: :obj:`256`)
    """
    idxs = torch.randperm(X.size(0))
    if X.is_cuda:
         idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        res = [X[batch_idxs]]
        for tensor in tensors:
            res.append(tensor[batch_idxs])
        yield res


from . import train_reach_angle_regressor
from . import train_sleep_classifier
