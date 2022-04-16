import torch

try:
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    _torch_geometric_available = False
else:
    _torch_geometric_available = True

from .relative_positioning_dataloader import RelativePositioningDataLoader


class RelativeSequenceDataLoader(RelativePositioningDataLoader):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True,
                 pos_kmin=0, pos_kmax=0, transform=None, **kwargs):
        super().__init__(dataset, batch_size, drop_last=drop_last, shuffle=shuffle,
                         pos_kmin=pos_kmin, pos_kmax=pos_kmax, transform=transform, **kwargs)

    def _add_pos_cand_edge_indices(self, data):
        n = data.num_nodes
        # positive edge index: default self only
        data.pos_edge_index = self._fast_diag(n, self.pos_kmin, self.pos_kmax)
        # complementary negative edge index
        data.ccand_edge_index, _ = dense_to_sparse(torch.ones((n, n)))
        return data
