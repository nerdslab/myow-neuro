import os
import pickle
import logging
from functools import lru_cache

import numpy as np
import torch
from tqdm import tqdm

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    _torch_geometric_available = False
else:
    _torch_geometric_available = True

from .utils import loadmat


class MonkeyReachNeuralDataset:
    FILENAMES = {
        ('mihi', 1): 'full-mihi-03032014',
        ('mihi', 2): 'full-mihi-03062014',
        ('chewie', 1): 'full-chewie-10032013',
        ('chewie', 2): 'full-chewie-12192013',
    }

    def __init__(self, root, primate='mihi', day=1, split='train', train_split=0.8, val_split=0.1,
                 binning_period=0.1, velocity_threshold=5.):
        if not _torch_geometric_available:
            raise ImportError('`MonkeyReachNeuralDataset` requires `torch_geometric`.')

        self.root = root
        assert primate in ['mihi', 'chewie']
        assert day in [1, 2]
        self.primate = primate
        self.day = day

        # get path to data
        self.filename = self.FILENAMES[(self.primate, day)]
        self.raw_path = os.path.join(self.root, 'raw', '%s.mat') % self.filename
        self.processed_path = os.path.join(self.root, 'processed/{}-bin{:.2f}-vel_thresh{:.2f}.pkl'.format(
            self.filename, binning_period, velocity_threshold))

        # get pre-processing parameters
        self.binning_period = binning_period
        self.velocity_threshold = velocity_threshold

        # load processed data or process data
        if not os.path.exists(self.processed_path):
            data_train_test = self._process()
        else:
            data_train_test = self._load_processed_data()

        # train/val split
        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split
        self.train_split = train_split
        self.val_split = val_split

        # split data
        self.data = self._split_train_test(data_train_test, split=split)

    def __getitem__(self, idx):
        # returns the data for one trial
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @property
    @lru_cache
    def _full_data(self):
        return Batch.from_data_list(self.data)

    def __getattr__(self, item):
        if item == '_full_data':
            raise ValueError('_full_data not defined properly.')
        # (x=x, pos=pos, vel=vel, acc=acc, force=force, y=y, t=t, edge_index=edge_index)
        if item in self._full_data.keys:
            return getattr(self._full_data, item)
        else:
            raise ValueError

    @property
    def angle(self):
        return (2 * np.pi / 8 * self.y).unsqueeze(-1)

    @property
    def num_samples(self):
        return self._full_data.num_nodes

    @property
    def num_trials(self):
        return len(self.data)

    def get_mean_std(self, feature):
        feature = 'x' if feature in ['firing_rates', 'fr'] else feature
        x = self._full_data[feature]
        return x.mean(dim=0), x.std(dim=0)

    def __repr__(self):
        return '{}(primate={}, day={}, split={})'.format(self.__class__.__name__, self.primate, self.day, self.split)

    def _process(self):
        logging.info('Preparing dataset: Binning data.')
        # load data
        mat_dict = loadmat(self.raw_path)

        # bin data
        data = self._bin_data(mat_dict)

        # convert to graphs
        data = self._convert_to_graphs(data)

        self._save_processed_data(data)
        return data

    def _bin_data(self, mat_dict):
        # load matrix
        trialtable = mat_dict['trial_table']
        neurons = mat_dict['out_struct']['units']
        pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        data_list = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                     'force': [], 'labels': [], 'sequence': []}
        for trial_id in tqdm(range(num_trials)):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            # grids= minT:(delT-TO):(maxT-delT);
            grid = np.arange(min_T, max_T + self.binning_period, self.binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids)

            neurons_binned = np.zeros((num_bins, num_neurons))
            pos_binned = np.zeros((num_bins, 2))
            vel_binned = np.zeros((num_bins, 2))
            acc_binned = np.zeros((num_bins, 2))
            force_binned = np.zeros((num_bins, 2))
            targets_binned = np.zeros((num_bins,))
            id_binned = np.arange(num_bins)

            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                if len(pos) > 0:
                    pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                vel_binned[k, :] = np.mean(vel[bin_mask, 1:], axis=0)
                if len(acc):
                    acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                if len(force) > 0:
                    force_binned[k, :] = np.mean(force[bin_mask, 1:], axis=0)
                targets_binned[k] = trialtable[trial_id, 1]

            for i in range(num_neurons):
                for k in range(num_bins):
                    spike_times = neurons[i]['ts']
                    bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                    neurons_binned[k, i] = np.sum(bin_mask) / self.binning_period

            # filter velocity
            mask = np.linalg.norm(vel_binned, 2, axis=1) > self.velocity_threshold

            data_list['firing_rates'].append(neurons_binned[mask])
            data_list['position'].append(pos_binned[mask])
            data_list['velocity'].append(vel_binned[mask])
            data_list['acceleration'].append(acc_binned[mask])
            data_list['force'].append(force_binned[mask])
            data_list['labels'].append(targets_binned[mask])
            data_list['sequence'].append(id_binned[mask])
        return data_list

    def _convert_to_graphs(self, data_list):
        num_trials = len(data_list['firing_rates'])
        graph_list = []

        for trial_id in range(num_trials):
            fr = torch.Tensor(data_list['firing_rates'][trial_id])
            pos = torch.Tensor(data_list['position'][trial_id])
            vel = torch.Tensor(data_list['velocity'][trial_id])
            acc = torch.Tensor(data_list['acceleration'][trial_id])
            force = torch.Tensor(data_list['force'][trial_id])
            y = torch.LongTensor(data_list['labels'][trial_id])
            timestep = torch.LongTensor(data_list['sequence'][trial_id])
            # build index
            edge_index, _ = dense_to_sparse(torch.diag(torch.ones(fr.size(0) - 1,), 1))
            # create graph
            graph = Data(x=fr, pos=pos, vel=vel, acc=acc, force=force, y=y, timestep=timestep, edge_index=edge_index)
            graph_list.append(graph)
        return graph_list

    def _save_processed_data(self, data):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        with open(self.processed_path, 'wb') as output:
            pickle.dump({'data': data}, output)
        logging.info('Processed data was saved to {}.'.format(self.processed_path))

    def _load_processed_data(self):
        logging.info('Loading processed data from {}.'.format(self.processed_path))
        with open(self.processed_path, "rb") as fp:
            data = pickle.load(fp)['data']
        return data

    def _split_train_test(self, data, split):
        if split is None:
            return data
        num_trials = len(data)
        split_id = int(num_trials * (self.train_split + self.val_split))

        if split == 'test':
            return data[split_id:]
        else:
            data = data[:split_id]
            if split == 'trainval':
                return data
            else:
                split_id = int(num_trials * self.train_split)
                if split == 'train':
                    return data[:split_id]
                elif split == 'val':
                    return data[split_id:]
