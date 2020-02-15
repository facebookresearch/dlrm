# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import torch
import time
import math


class DataLoader:
    """
    DataLoader dedicated for the Criteo Terabyte Click Logs dataset
    """

    def __init__(
            self,
            data_filename,
            data_directory,
            days,
            batch_size,
            max_ind_range = -1,
            split = "train",
            drop_last_batch=True
    ):
        self.data_filename = data_filename
        self.data_directory = data_directory
        self.days = days
        self.batch_size = batch_size

        total_file = os.path.join(
            data_directory,
            data_filename + "_day_count.npz"
        )
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"][np.array(days)]

        self.length = sum(total_per_file)
        if split == "test" or split == "val":
            self.length = int(np.ceil(self.length / 2.))
        self.split = split
        self.drop_last_batch = drop_last_batch
        self.max_ind_range = max_ind_range

    def __iter__(self):
        return iter(_batch_generator(self.data_filename, self.data_directory, self.days,
                                     self.batch_size, self.split, self.drop_last_batch, self.max_ind_range))

    def __len__(self):
        if self.drop_last_batch:
            return self.length // self.batch_size
        else:
            return math.ceil(self.length / self.batch_size)


def _batch_generator(data_filename, data_directory, days, batch_size, split, drop_last, max_ind_range):
    previous_file = None
    for day in days:
        filepath = os.path.join(
            data_directory,
            data_filename + "_{}_reordered.npz".format(day)
        )

        # print('Loading file: ', filepath)
        with np.load(filepath) as data:
            x_int = data["X_int"]
            x_cat = data["X_cat"]
            y = data["y"]

        samples_in_file = y.shape[0]
        batch_start_idx = 0
        if split == "test" or split == "val":
            length = int(np.ceil(samples_in_file / 2.))
            if split == "test":
                samples_in_file = length
            elif split == "val":
                batch_start_idx = samples_in_file - length

        while batch_start_idx < samples_in_file - batch_size:

            missing_samples = batch_size
            if previous_file is not None:
                missing_samples -= previous_file['y'].shape[0]

            current_slice = slice(batch_start_idx, batch_start_idx + missing_samples)

            x_int_batch = x_int[current_slice]
            x_cat_batch = x_cat[current_slice]
            y_batch = y[current_slice]

            if previous_file is not None:
                x_int_batch = np.concatenate(
                    [previous_file['x_int'], x_int_batch],
                    axis=0
                )
                x_cat_batch = np.concatenate(
                    [previous_file['x_cat'], x_cat_batch],
                    axis=0
                )
                y_batch = np.concatenate([previous_file['y'], y_batch], axis=0)
                previous_file = None


            if x_int_batch.shape[0] != batch_size:
                raise ValueError('should not happen')

            yield _transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range)

            batch_start_idx += missing_samples
        if batch_start_idx != samples_in_file:
            current_slice = slice(batch_start_idx, samples_in_file)
            if previous_file is not None:
                previous_file = {
                    'x_int' : np.concatenate(
                        [previous_file['x_int'], x_int[current_slice]],
                        axis=0
                    ),
                    'x_cat' : np.concatenate(
                        [previous_file['x_cat'], x_cat[current_slice]],
                        axis=0
                    ),
                    'y' : np.concatenate([previous_file['y'], y[current_slice]], axis=0)
                }
            else:
                previous_file = {
                    'x_int' : x_int[current_slice],
                    'x_cat' : x_cat[current_slice],
                    'y' : y[current_slice]
                }

    if not drop_last:
        yield _transform_features(previous_file['x_int'],
                                  previous_file['x_cat'],
                                  previous_file['y'], max_ind_range)


def _transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range):
    if max_ind_range > 0: x_cat_batch = x_cat_batch % max_ind_range
    x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
    x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
    y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)


def _test():
    generator = _batch_generator(
        data_filename='day',
        data_directory='/input',
        days=range(23),
        split="train",
        batch_size=2048
    )
    t1 = time.time()
    for x_int, lS_o, x_cat, y in generator:
        t2 = time.time()
        time_diff = t2 - t1
        t1 = t2
        print(
            "time {} x_int.shape: {} lS_o.shape: {} x_cat.shape: {} y.shape: {}".format(
                time_diff, x_int.shape, lS_o.shape, x_cat.shape, y.shape
            )
        )


if __name__ == '__main__':
    _test()
