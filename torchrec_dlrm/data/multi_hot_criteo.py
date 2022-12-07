#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from iopath.common.file_io import PathManager, PathManagerFactory
from pyre_extensions import none_throws
from torchrec.datasets.utils import (
    Batch,
    PATH_MANAGER_KEY,
)
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class MultiHotCriteoIterDataPipe(InMemoryBinaryCriteoIterDataPipe):
    """
    Datapipe designed to operate over binary (npy) versions of Criteo datasets. Loads
    the entire dataset into memory to prevent disk speed from affecting throughout. Each
    rank reads only the data for the portion of the dataset it is responsible for.

    The torchrec/datasets/scripts/npy_preproc_criteo.py script can be used to convert
    the Criteo tsv files to the npy files expected by this dataset.

    Args:
        stage (str): "train", "val", or "test".
        dense_paths (List[str]): List of path strings to dense npy files.
        sparse_paths (List[str]): List of path strings to multi-hot sparse npz files.
        labels_paths (List[str]): List of path strings to labels npy files.
        batch_size (int): batch size.
        rank (int): rank.
        world_size (int): world size.
        shuffle_batches (bool): Whether to shuffle batches
        hashes (Optional[int]): List of max categorical feature value for each feature.
            Length of this list should be CAT_FEATURE_COUNT.
        path_manager_key (str): Path manager key used to load from different
            filesystems.

    Example::

        template = "/home/datasets/criteo/1tb_binary/day_{}_{}.npy"
        datapipe = InMemoryBinaryCriteoIterDataPipe(
            dense_paths=[template.format(0, "dense"), template.format(1, "dense")],
            sparse_paths=[template.format(0, "sparse"), template.format(1, "sparse")],
            labels_paths=[template.format(0, "labels"), template.format(1, "labels")],
            batch_size=1024,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        batch = next(iter(datapipe))
    """

    def __init__(
        self,
        stage: str,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle_batches: bool = False,
        shuffle_training_set: bool = False,
        shuffle_training_set_random_seed: int = 0,
        mmap_mode: bool = False,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        self.stage = stage
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.shuffle_training_set = shuffle_training_set
        np.random.seed(shuffle_training_set_random_seed)
        self.mmap_mode = mmap_mode
        self.hashes: np.ndarray = np.array(hashes).reshape((1, CAT_FEATURE_COUNT))
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)

        if shuffle_training_set and stage == "train":
            self._shuffle_and_load_data_for_rank()
        else:
            self._load_data_for_rank()
        # When mmap_mode is enabled, sparse features are hashed when
        # samples are batched in def __iter__. Otherwise, the dataset has been
        # preloaded with sparse features hashed in the preload stage, here:
        if not self.mmap_mode and self.hashes is not None:
            self.sparse_arrs = [fts % hash for (fts, hash) in zip(self.sparse_arrs, self.hashes)]

        self.num_rows_per_file: List[int] = [a.shape[0] for a in self.dense_arrs]
        cur_rank_dataset_len = sum(self.num_rows_per_file)
        if self.rank < self.remainder:
            self.num_batches: int = math.ceil((cur_rank_dataset_len - 1) / batch_size)
        else:
            self.num_batches: int = math.ceil(cur_rank_dataset_len / batch_size)

        self.multi_hot_sizes: List[int] = [
            multi_hot_feat.shape[-1] for multi_hot_feat in self.sparse_arrs[0]
        ]

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def _np_arrays_to_batch(
        self,
        dense: np.ndarray,
        sparse: List[np.ndarray],
        labels: np.ndarray,
    ) -> Batch:
        if self.shuffle_batches:
            # Shuffle all 3 in unison
            shuffler = np.random.permutation(len(dense))
            sparse = [multi_hot_ft[shuffler, :] for multi_hot_ft in sparse]
            dense = dense[shuffler]
            labels = labels[shuffler]

        batch_size = len(dense)
        lengths = torch.ones((CAT_FEATURE_COUNT * batch_size), dtype=torch.int32)
        for k, multi_hot_size in enumerate(self.multi_hot_sizes):
            lengths[k * batch_size : (k + 1) * batch_size] = multi_hot_size
        offsets = torch.cumsum(torch.concat((torch.tensor([0]), lengths)), dim=0)
        length_per_key = [
            batch_size * multi_hot_size for multi_hot_size in self.multi_hot_sizes
        ]
        offset_per_key = torch.cumsum(
            torch.concat((torch.tensor([0]), torch.tensor(length_per_key))), dim=0
        )
        values = torch.concat(sparse)
        return Batch(
            dense_features=torch.from_numpy(dense.copy()),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                values=values,
                lengths=lengths,
                offsets=offsets,
                stride=batch_size,
                length_per_key=length_per_key,
                offset_per_key=offset_per_key.tolist(),
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1).copy()),
        )

    def __iter__(self) -> Iterator[Batch]:
        # Invariant: buffer never contains more than batch_size rows.
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(
            dense: np.ndarray,
            sparse: List[np.ndarray],
            labels: np.ndarray,
        ) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [dense, sparse, labels]
            else:
                buffer[0] = np.concatenate((buffer[0], dense))
                buffer[1] = list(map(np.concatenate, zip(buffer[1], sparse)))
                buffer[2] = np.concatenate((buffer[2], labels))

        # Maintain a buffer that can contain up to batch_size rows. Fill buffer as
        # much as possible on each iteration. Only return a new batch when batch_size
        # rows are filled.
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        cur_batch_size = self.batch_size
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if buffer is None else none_throws(buffer)[0].shape[0]
            if buffer_row_count == cur_batch_size or file_idx == len(self.dense_arrs):
                yield self._np_arrays_to_batch(*none_throws(buffer))
                batch_idx += 1
                buffer = None
                if batch_idx + 1 == self.num_batches and self.rank < self.remainder:
                    cur_batch_size += 1
            else:
                rows_to_get = min(
                    cur_batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                slice_ = slice(row_idx, row_idx + rows_to_get)

                sparse_inputs = [feats[slice_, :] for feats in self.sparse_arrs[file_idx]]
                dense_inputs = self.dense_arrs[file_idx][slice_, :]
                target_labels = self.labels_arrs[file_idx][slice_, :]

                if self.mmap_mode and self.hashes is not None:
                    sparse_inputs = [fts % hash for (fts, hash) in zip(sparse_inputs, self.hashes)]

                append_to_buffer(
                    dense_inputs,
                    sparse_inputs,
                    target_labels,
                )
                row_idx += rows_to_get

                if row_idx >= self.num_rows_per_file[file_idx]:
                    file_idx += 1
                    row_idx = 0

    def __len__(self) -> int:
        return self.num_batches
