import os

import numpy as np
import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    _torch_geometric_available = False
else:
    _torch_geometric_available = True


class RodentNeuralDataset:
    FILENAMES = {
        'mouse': ('fr_bins_n_4full.npy', 'behav_states_n_4full.npy'),
        'rat': ('XYF06_1217_b5_neurons_qualityfr_bins_n_4full.npy', 'XYF06_1217_b5_neurons_qualitybehav_states_n_4full.npy'),
    }

    def __init__(self, root, rodent='mouse', split='train', train_split=0.7, val_split=0.1):
        if not _torch_geometric_available:
            raise ImportError('`RodentNeuralDataset` requires `torch_geometric`.')

        self.root = root
        # get path to data
        assert rodent in ['mouse', 'rat']
        self.rodent = rodent

        self.fr_filename, self.label_filename = self.FILENAMES[self.rodent]
        self.fr_path = os.path.join(self.root, self.fr_filename)
        self.label_path = os.path.join(self.root, self.label_filename)

        # load data
        data_train_test = self._load()
        data_train_test = self._convert_to_graph(data_train_test)

        # train/val split
        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split
        self.train_split = train_split
        self.val_split = val_split

        # split data
        self.data = self._split_train_test(data_train_test, split=split)

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.data

    def __getattr__(self, item):
        # (x=x, y=y, t=t, edge_index=edge_index)
        if item in self.data.keys:
            return getattr(self.data, item)
        else:
            raise AttributeError

    @property
    def num_samples(self):
        return self.data.num_nodes

    def get_mean_std(self, feature):
        feature = 'x' if feature in ['firing_rates', 'fr'] else feature
        x = self.data[feature]
        return x.mean(dim=0), x.std(dim=0)

    def __repr__(self):
        return '{}(rodent={}, split={})'.format(self.__class__.__name__, self.rodent, self.split)

    def _load(self):
        firing_rates = np.load(self.fr_path, allow_pickle=True).T
        labels = np.load(self.label_path, allow_pickle=True)
        return {'firing_rates': firing_rates, 'labels': labels}

    def _convert_to_graph(self, data):
        x = torch.Tensor(data['firing_rates'])
        y = torch.LongTensor(data['labels'])
        t = torch.arange(0, y.size(0), dtype=torch.long)

        # build index
        edge_index, _ = dense_to_sparse(torch.diag(torch.ones(x.size(0) - 1,), 1))
        # create graph
        graph = Data(x=x, y=y, timestep=t, edge_index=edge_index)
        return graph

    def _split_train_test(self, data, split):
        if split is None:
            return data

        num_samples = len(data.y)
        split_id = int(num_samples * (self.train_split + self.val_split))

        sub_data = Data()

        if split == 'test':
            for key, item in data:
                sub_data[key] = item[split_id:]
            sub_data.edge_index, _ = dense_to_sparse(torch.diag(torch.ones(sub_data.x.size(0) - 1,), 1))
            return sub_data
        else:
            for key, item in data:
                data[key] = item[:split_id]
            if split == 'trainval':
                data.edge_index, _ = dense_to_sparse(torch.diag(torch.ones(data.x.size(0) - 1, ), 1))
                return data
            else:
                split_id = int(len(data.y) * self.train_split)
                if split == 'train':
                    for key, item in data:
                        sub_data[key] = item[:split_id]
                elif split == 'val':
                    for key, item in data:
                        sub_data[key] = item[split_id:]
                sub_data.edge_index, _ = dense_to_sparse(torch.diag(torch.ones(sub_data.x.size(0) - 1, ), 1))
                return sub_data
