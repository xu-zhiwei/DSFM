import pickle
from pathlib import Path
from queue import Queue

import torch
from torch.utils.data import Dataset, DataLoader


class Vocab:
    def __init__(self, data_root):
        self.data_root = Path(data_root)

        subtrees_list = []
        with open(self.data_root / 'subtrees.pkl', 'rb') as f:
            while True:
                try:
                    subtrees_list.append(pickle.load(f))
                except EOFError:
                    break

        self.token_to_idx = {'[PAD]': 0, '[OOV]': 1}
        self.processed = set()
        with open(self.data_root / 'train.pkl', 'rb') as f:
            while True:
                try:
                    (idx1, idx2), _ = pickle.load(f)
                    for idx in (idx1, idx2):
                        if idx in self.processed:
                            continue
                        subtrees = subtrees_list[idx]
                        for subtree in subtrees:
                            q = Queue()
                            q.put(subtree)
                            while not q.empty():
                                st = q.get()
                                for obj in st:
                                    if isinstance(obj, list):
                                        q.put(obj)
                                    else:
                                        if obj in self.token_to_idx:
                                            continue
                                        self.token_to_idx[obj] = len(self.token_to_idx)
                    self.processed.add(idx)
                except EOFError:
                    break

    @staticmethod
    def get(v, q):
        return v[q] if q in v else v['[OOV]']


class SubtreeDataset(Dataset):
    def __init__(self, data_root, vocab):
        self.data_root = Path(data_root)
        self.vocab = vocab

        def dfs(obj):
            if isinstance(obj, list):
                ret = []
                for o in obj:
                    ret.append(dfs(o))
                return ret
            else:
                return Vocab.get(self.vocab.token_to_idx, obj)

        self.features = []
        with open(self.data_root / 'subtrees.pkl', 'rb') as f:
            while True:
                try:
                    subtrees = pickle.load(f)
                    feature = []
                    for subtree in subtrees:
                        feature.append(dfs(subtree))
                    self.features.append(feature)
                except EOFError:
                    break

    def __getitem__(self, idx):
        return self.features[idx]


class PairDataset(Dataset):
    def __init__(self, data_root, data_role):
        self.data_root = Path(data_root)
        self.data_role = data_role

        self.pairs, self.labels = [], []
        with open(self.data_root / f'{self.data_role}.pkl', 'rb') as fp:
            while True:
                try:
                    (idx1, idx2), label = pickle.load(fp)
                    self.pairs.append((idx1, idx2))
                    self.labels.append(label)
                except EOFError:
                    break

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class Collator:
    def __init__(self, data_root, vocab):
        self.dataset = SubtreeDataset(data_root, vocab)

    def __call__(self, batch):
        subtrees1 = []
        subtrees2 = []
        pairs, labels = zip(*batch)
        for idx1, idx2 in pairs:
            subtrees1.append(self.dataset[idx1])
            subtrees2.append(self.dataset[idx2])

        labels = torch.FloatTensor(labels)
        return (subtrees1, subtrees2), labels


class Data:
    def __init__(self, data_root, batch_size=128, num_workers=0):
        self.data_root = Path(data_root)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab = Vocab(self.data_root)
        self.collator = Collator(self.data_root, self.vocab)

        self.train_dataset = PairDataset(self.data_root, 'train')
        self.val_dataset = PairDataset(self.data_root, 'val')
        self.test_dataset = PairDataset(self.data_root, 'test')

    def train_loader(self):
        return DataLoader(self.train_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_loader(self):
        return DataLoader(self.test_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers)
