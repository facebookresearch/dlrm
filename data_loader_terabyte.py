# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import argparse
import math

_features_per_sample = 40
_bytes_per_feature = 4


def _transform_features(x_int_batch, x_cat_batch, y_batch):
    x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
    x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
    y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)


def numpy_to_binary(input_files, output_file_path, split='train'):
    """Convert the data to a binary format to be read with CriteoBinDataset."""

    with open(output_file_path, 'wb') as output_file:
        if split == 'train':
            for input_file in input_files:
                print('Processing file: ', input_file)

                np_data = np.load(input_file)
                np_data = np.concatenate([np_data['y'].reshape(-1, 1),
                                          np_data['X_int'],
                                          np_data['X_cat']], axis=1)
                np_data = np_data.astype(np.float32)

                output_file.write(np_data.tobytes())
        else:
            assert len(input_files) == 1
            np_data = np.load(input_files[0])
            np_data = np.concatenate([np_data['y'].reshape(-1, 1),
                                      np_data['X_int'],
                                      np_data['X_cat']], axis=1)
            np_data = np_data.astype(np.float32)

            samples_in_file = np_data.shape[0]
            midpoint = int(np.ceil(samples_in_file / 2.))
            if split == "test":
                begin = 0
                end = midpoint
            elif split == "val":
                begin = midpoint
                end = samples_in_file
            else:
                raise ValueError('Unknown split value: ', split)

            output_file.write(np_data[begin:end].tobytes())


class CriteoBinDataset(Dataset):
    """Binary version of criteo dataset."""

    def __init__(self, data_file, counts_file, batch_size=1):
        self.batch_size = batch_size
        self.bytes_per_entry = (_bytes_per_feature *
                                _features_per_sample *
                                batch_size)

        self.num_entries = math.ceil(os.path.getsize(data_file) /
                                     self.bytes_per_entry)

        print('data file:', data_file, 'number of batches:', self.num_entries)
        self.file = open(data_file, 'rb')

        with np.load(counts_file) as data:
            self.counts = data["counts"]

        # hardcoded for now
        self.m_den = 13

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        self.file.seek(idx * self.bytes_per_entry, 0)
        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=np.float32)
        tensor = torch.from_numpy(array).view((-1, _features_per_sample))

        return _transform_features(x_int_batch=tensor[:,1:14],
                                   x_cat_batch=tensor[:,14:],
                                   y_batch=tensor[:,0])


def _preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', required=True)
    parser.add_argument('--input_data_prefix', required=True)
    args = parser.parse_args()

    train_files = ['{}_{}_reordered.npz'.format(args.input_data_prefix,day) for
                   day in range(0,23)]

    test_valid_file =  args.input_data_prefix + '_23_reordered.npz'

    os.makedirs(args.output_directory, exist_ok=True)
    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        output_file = os.path.join(args.output_directory,
                                   '{}_data.bin'.format(split))

        input_files = train_files if split == 'train' else [test_valid_file]
        numpy_to_binary(input_files=input_files,
                        output_file_path=output_file,
                        split=split)


def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', required=True)
    parser.add_argument('--input_data_prefix', required=True)
    args = parser.parse_args()

    train_file = os.path.join(args.output_directory, 'train_data.bin')
    counts_file = os.path.join(args.output_directory, 'day_fea_count.npz')
    train_dataset_binary = CriteoBinDataset(data_file=train_file,
                                            counts_file=counts_file,
                                            batch_size=2048,)
    from dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo

    binary_loader = torch.utils.data.DataLoader(
        train_dataset_binary,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )

    original_train_dataset = CriteoDataset(
        dataset='terabyte',
        max_ind_range=10 * 1000 * 1000,
        sub_sample_rate=1,
        randomize=True,
        split="train",
        raw_path=args.input_data_prefix,
        pro_data='dummy_string',
        memory_map=True
    )

    original_train_loader = torch.utils.data.DataLoader(
        original_train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataset_binary) == len(original_train_loader)
    for i, (old_batch, new_batch) in enumerate(zip(original_train_loader,
                                               binary_loader)):

        for j in range(len(new_batch)):
            if not np.array_equal(old_batch[j], new_batch[j]):
                raise ValueError('FAILED: Datasets not equal')
        if i > len(train_dataset_binary):
            break
    print('PASSED')


if __name__ == '__main__':
    _preprocess()
    #_test()
