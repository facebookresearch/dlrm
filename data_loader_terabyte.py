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
            drop_last_batch=False
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
        self.drop_last_batch = drop_last_batch

    def __iter__(self):
        return iter(_batch_generator(self.data_filename, self.data_directory, self.days,
                                     self.batch_size, self.drop_last_batch))

    def __len__(self):
        if self.drop_last_batch:
            return math.ceil(self.length / self.batch_size)
        else:
            return self.length // self.batch_size


def _batch_generator(data_filename, data_directory, days, batch_size, drop_last_batch):
    previous_file = None
    for day in days:
        filepath = os.path.join(
            data_directory,
            data_filename + "_{}_reordered.npz".format(day)
        )

        print('Loading file: ', filepath)
        with np.load(filepath) as data:
            x_int = data["X_int"]
            x_cat = data["X_cat"]
            y = data["y"]

        samples_in_file = y.shape[0]

        batch_start_idx = 0
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

            yield _transform_features(x_int_batch, x_cat_batch, y_batch)

            batch_start_idx += missing_samples
        if batch_start_idx != samples_in_file:
            current_slice = slice(batch_start_idx, samples_in_file)
            previous_file = {
                'x_int' : x_int[current_slice],
                'x_cat' : x_cat[current_slice],
                'y' : y[current_slice]
            }

    if not drop_last_batch:
        # print('last batch!')
        yield _transform_features(previous_file['x_int'],
                                  previous_file['x_cat'],
                                  previous_file['y'])


def _transform_features(x_int_batch, x_cat_batch, y_batch):
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
        batch_size=2048
    )
    t1 = time.time()
    for x_int, lS_o, x_cat, y in generator:
        t2 = time.time()
        time_diff = t2 - t1
        t1 = t2
        print(
            "time: {} x_int shape: {} lS_o.shape: {} x_cat.shape: {} y.shape: {}".format(
                time_diff, x_int.shape, lS_o.shape, x_cat.shape, y.shape
            )
        )


if __name__ == '__main__':
    _test()
