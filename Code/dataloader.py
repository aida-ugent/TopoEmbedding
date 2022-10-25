import os

import numpy as np
from torch import DoubleTensor, LongTensor, hstack
from torch.utils.data import Dataset, DataLoader


class NetworkDataset(Dataset):
    def __init__(self, A, sample_ratio=10):
        self.m, self.n = A.shape
        self.indices = A.indices
        self.indptr = A.indptr
        self.sample_ratio = sample_ratio

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        nbr_ids = self.indices[self.indptr[idx]:self.indptr[idx + 1]]
        neg_samples = list(set(np.random.randint(self.n, size=(
            (len(nbr_ids) or 1) * self.sample_ratio,))) - set(nbr_ids) - set([idx]))

        items = np.hstack((nbr_ids, neg_samples))
        users = [idx] * len(items)

        target = np.zeros_like(items)
        target[:len(nbr_ids)] = 1

        return LongTensor(users), LongTensor(items), DoubleTensor(target)


def collate_fn(batch):
    u = hstack([entry[0] for entry in batch])
    v = hstack([entry[1] for entry in batch])
    label = hstack([entry[2] for entry in batch])
    return u, v, label


def config_network_loader(A, batch_size=None, num_workers=0):
    return DataLoader(NetworkDataset(A),
                      batch_size=batch_size or A.shape[0],
                      collate_fn=collate_fn,
                      num_workers=num_workers or os.cpu_count())


def config_edge_loader(E, batch_size=None, num_workers=0):
    return DataLoader(LongTensor(E),
                      batch_size=batch_size or len(E),
                      collate_fn=collate_fn,
                      num_workers=num_workers or os.cpu_count())


def dot_to_dictionary(config):
    dict_config = dict()
    param_keys, param_values = config.keys(), list(config.values())

    # Transform from dot.notation to {"dot": notation}
    for i, key in enumerate(param_keys):
        split_key = str.split(key, sep=".")
        if len(split_key) > 1:
            subkey = split_key[0]
            if subkey not in dict_config:
                dict_config[subkey] = {split_key[1]: param_values[i]}
            else:
                dict_config[subkey][split_key[1]] = param_values[i]
        else:
            dict_config[key] = param_values[i]

    return dict_config
