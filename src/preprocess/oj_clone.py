import os
import pickle
from pathlib import Path


def generate_pairs(source_data_dir, target_data_dir):
    def obtain_filepath(path):
        category, file = path.split('/')[-2:]
        return os.path.join(source_data_dir, 'OnlineJudge', category, file)

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    path2idx = {}
    with open(os.path.join(target_data_dir, 'codes.pkl'), 'wb') as fc:
        for part in ('train', 'val', 'test'):
            with open(os.path.join(source_data_dir, f'{part}data.txt'), 'r', encoding='utf-8') as f, \
                    open(os.path.join(target_data_dir, f'{part}.pkl'), 'wb') as fp:
                for line in f:
                    sample = line.strip().split('\t')
                    path1 = obtain_filepath(sample[0])
                    path2 = obtain_filepath(sample[1])
                    label = 1 if sample[2] == '1' else 0

                    for p in (path1, path2):
                        if p not in path2idx:
                            path2idx[p] = len(path2idx)
                            with open(p, 'r', encoding='utf-8') as fr:
                                pickle.dump(fr.read(), fc)
                    idx1, idx2 = path2idx[path1], path2idx[path2]
                    pickle.dump(((idx1, idx2), label), fp)
