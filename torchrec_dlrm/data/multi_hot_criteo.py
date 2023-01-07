#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import zipfile
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from iopath.common.file_io import PathManager, PathManagerFactory
from pyre_extensions import none_throws
from torch.utils.data import IterableDataset
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
)
from torchrec.datasets.utils import Batch, PATH_MANAGER_KEY
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class MultiHotCriteoIterDataPipe(IterableDataset):
    """
    Datapipe designed to operate over the MLPerf DLRM v2 synthetic multi-hot dataset.
    This dataset can be created by following the steps in
    torchrec_dlrm/scripts/materialize_synthetic_multihot_dataset.py.
    Each rank reads only the data for the portion of the dataset it is responsible for.

    Args:
        stage (str): "train", "val", or "test".
        dense_paths (List[str]): List of path strings to dense npy files.
        sparse_paths (List[str]): List of path strings to multi-hot sparse npz files.
        labels_paths (List[str]): List of path strings to labels npy files.
        batch_size (int): batch size.
        rank (int): rank.
        world_size (int): world size.
        drop_last (Optional[bool]): Whether to drop the last batch if it is incomplete.
        shuffle_batches (bool): Whether to shuffle batches
        shuffle_training_set (bool): Whether to shuffle all samples in the dataset.
        shuffle_training_set_random_seed (int): The random generator seed used when
            shuffling the training set.
        hashes (Optional[int]): List of max categorical feature value for each feature.
            Length of this list should be CAT_FEATURE_COUNT.
        path_manager_key (str): Path manager key used to load from different
            filesystems.

    Example::

        datapipe = MultiHotCriteoIterDataPipe(
            dense_paths=["day_0_dense.npy"],
            sparse_paths=["day_0_sparse_multi_hot.npz"],
            labels_paths=["day_0_labels.npy"],
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
        drop_last: Optional[bool] = False,
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
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.shuffle_training_set = shuffle_training_set
        np.random.seed(shuffle_training_set_random_seed)
        self.mmap_mode = mmap_mode
        self.hashes: np.ndarray = np.array(hashes).reshape((CAT_FEATURE_COUNT, 1))
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)

        if shuffle_training_set and stage == "train":
            # Currently not implemented for the materialized multi-hot dataset.
            self._shuffle_and_load_data_for_rank()
        else:
            m = "r" if mmap_mode else None
            self.dense_arrs: List[np.ndarray] = [
                np.load(f, mmap_mode=m) for f in self.dense_paths
            ]
            self.labels_arrs: List[np.ndarray] = [
                np.load(f, mmap_mode=m) for f in self.labels_paths
            ]
            self.sparse_arrs: List = []
            for sparse_path in self.sparse_paths:
                multi_hot_ids_l = []
                for feat_id_num in range(CAT_FEATURE_COUNT):
                    multi_hot_ft_ids = self._load_from_npz(
                        sparse_path, f"{feat_id_num}.npy"
                    )
                    multi_hot_ids_l.append(multi_hot_ft_ids)
                self.sparse_arrs.append(multi_hot_ids_l)
        d0len = len(self.dense_arrs[0])
        second_half_start_index = int(d0len // 2 + d0len % 2)
        if stage == "val":
            self.dense_arrs[0] = self.dense_arrs[0][:second_half_start_index, :]
            self.labels_arrs[0] = self.labels_arrs[0][:second_half_start_index, :]
            self.sparse_arrs[0] = [
                feats[:second_half_start_index, :] for feats in self.sparse_arrs[0]
            ]
        elif stage == "test":
            self.dense_arrs[0] = self.dense_arrs[0][second_half_start_index:, :]
            self.labels_arrs[0] = self.labels_arrs[0][second_half_start_index:, :]
            self.sparse_arrs[0] = [
                feats[second_half_start_index:, :] for feats in self.sparse_arrs[0]
            ]
        # When mmap_mode is enabled, sparse features are hashed when
        # samples are batched in def __iter__. Otherwise, the dataset has been
        # preloaded with sparse features hashed in the preload stage, here:
        if not self.mmap_mode and self.hashes is not None:
            for k, _ in enumerate(self.sparse_arrs):
                self.sparse_arrs[k] = [
                    feat % hash
                    for (feat, hash) in zip(self.sparse_arrs[k], self.hashes)
                ]

        self.num_rows_per_file: List[int] = list(map(len, self.dense_arrs))
        total_rows = sum(self.num_rows_per_file)
        global_batch_size = self.world_size * batch_size
        self.num_batches: int = total_rows // global_batch_size
        self.local_rank_num_batches: int = total_rows // global_batch_size
        if self.local_rank_num_batches == 0:
            self.batch_size = batch_size = total_rows // self.world_size
            self.local_rank_num_batches = 1

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

    def _load_from_npz(self, fname, npy_name):
        # figure out offset of .npy in .npz
        zf = zipfile.ZipFile(fname)
        info = zf.NameToInfo[npy_name]
        assert info.compress_type == 0
        zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)
        # read .npy header
        zf.open(npy_name, "r")
        version = np.lib.format.read_magic(zf.fp)
        shape, fortran_order, dtype = np.lib.format._read_array_header(zf.fp, version)
        assert (
            dtype == "int32"
        ), f"sparse multi-hot dtype is {dtype} but should be int32"
        offset = zf.fp.tell()
        # create memmap
        return np.memmap(
            zf.filename,
            dtype=dtype,
            shape=shape,
            order="F" if fortran_order else "C",
            mode="r",
            offset=offset,
        )

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
        values = torch.concat([torch.from_numpy(feat).flatten() for feat in sparse])
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
                buffer[1] = [np.concatenate((b, s)) for b, s in zip(buffer[1], sparse)]
                buffer[2] = np.concatenate((buffer[2], labels))

        # Maintain a buffer that can contain up to batch_size rows. Fill buffer as
        # much as possible on each iteration. Only return a new batch when batch_size
        # rows are filled.
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        buffer_row_count = 0
        while batch_idx // self.world_size < self.local_rank_num_batches + int(
            not self.drop_last
        ):
            if buffer_row_count == self.batch_size or file_idx == len(self.dense_arrs):
                local_batch_idx = batch_idx // self.world_size
                if batch_idx % self.world_size == self.rank and (
                    not self.drop_last
                    and local_batch_idx != self.local_rank_num_batches - 1
                    or self.drop_last
                ):
                    yield self._np_arrays_to_batch(*none_throws(buffer))
                    buffer = None
                buffer_row_count = 0
                batch_idx += 1
            else:
                rows_to_get = min(
                    self.batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                buffer_row_count += rows_to_get
                slice_ = slice(row_idx, row_idx + rows_to_get)

                if batch_idx % self.world_size == self.rank:
                    dense_inputs = self.dense_arrs[file_idx][slice_, :]
                    sparse_inputs = [
                        feats[slice_, :] for feats in self.sparse_arrs[file_idx]
                    ]
                    target_labels = self.labels_arrs[file_idx][slice_, :]

                    if self.mmap_mode and self.hashes is not None:
                        sparse_inputs = [
                            feats % hash
                            for (feats, hash) in zip(sparse_inputs, self.hashes)
                        ]

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
