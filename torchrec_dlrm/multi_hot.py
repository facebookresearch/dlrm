# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np

import torch
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class RestartableMap:
    def __init__(self, f, source):
        self.source = source
        self.func = f

    def __iter__(self):
        for x in self.source:
            yield self.func(x)

    def __len__(self):
        return len(self.source)


class Multihot:
    def __init__(
        self,
        multi_hot_sizes: List[int],
        num_embeddings_per_feature: List[int],
        batch_size: int,
        collect_freqs_stats: bool,
        dist_type: str = "uniform",
    ):
        if dist_type not in {"uniform", "pareto"}:
            raise ValueError(
                "Multi-hot distribution type {} is not supported."
                'Only "uniform" and "pareto" are supported.'.format(dist_type)
            )
        self.dist_type = dist_type
        self.multi_hot_sizes = multi_hot_sizes
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.batch_size = batch_size

        # Generate 1-hot to multi-hot lookup tables, one lookup table per sparse embedding table.
        self.multi_hot_tables_l = self.__make_multi_hot_indices_tables(
            dist_type, multi_hot_sizes, num_embeddings_per_feature
        )

        # Pooling offsets are computed once and reused.
        self.offsets = self.__make_offsets(multi_hot_sizes, num_embeddings_per_feature, batch_size)

        # For plotting frequency access
        self.collect_freqs_stats = collect_freqs_stats
        self.model_to_track = None
        self.freqs_pre_hash = []
        self.freqs_post_hash = []
        for embs_count in num_embeddings_per_feature:
            self.freqs_pre_hash.append(np.zeros((embs_count)))
            self.freqs_post_hash.append(np.zeros((embs_count)))

    def save_freqs_stats(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        pre_dict = {str(k): e for k, e in enumerate(self.freqs_pre_hash)}
        np.save(f"stats_pre_hash_{rank}_{self.dist_type}.npy", pre_dict)
        post_dict = {str(k): e for k, e in enumerate(self.freqs_post_hash)}
        np.save(f"stats_post_hash_{rank}_{self.dist_type}.npy", post_dict)

    def pause_stats_collection_during_val_and_test(self, model: torch.nn.Module) -> None:
        self.model_to_track = model

    def __make_multi_hot_indices_tables(
        self,
        dist_type: str,
        multi_hot_sizes: List[int],
        num_embeddings_per_feature: List[int],
    ) -> List[np.array]:
        np.random.seed(0)  # The seed is necessary for all ranks to produce the same lookup values.
        multi_hot_tables_l = []
        for embs_count, multi_hot_size in zip(num_embeddings_per_feature, multi_hot_sizes):
            embedding_ids = np.arange(embs_count)[:, np.newaxis]
            if dist_type == "uniform":
                synthetic_sparse_ids = np.random.randint(
                    0, embs_count, size=(embs_count, multi_hot_size - 1)
                )
            elif dist_type == "pareto":
                synthetic_sparse_ids = (
                    np.random.pareto(a=0.25, size=(embs_count, multi_hot_size - 1)).astype(np.int32)
                    % embs_count
                )
            multi_hot_table = np.concatenate((embedding_ids, synthetic_sparse_ids), axis=-1)
            multi_hot_tables_l.append(multi_hot_table)
        multi_hot_tables_l = [
            torch.from_numpy(multi_hot_table).int() for multi_hot_table in multi_hot_tables_l
        ]
        return multi_hot_tables_l

    def __make_offsets(
        self,
        multi_hot_sizes: int,
        num_embeddings_per_feature: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        lS_o = torch.ones((len(num_embeddings_per_feature) * batch_size), dtype=torch.int32)
        for k, multi_hot_size in enumerate(multi_hot_sizes):
            lS_o[k * batch_size : (k + 1) * batch_size] = multi_hot_size
        lS_o = torch.cumsum(torch.concat((torch.tensor([0]), lS_o)), axis=0)
        return lS_o

    def __make_new_batch(
        self,
        lS_i: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lS_i = lS_i.reshape(-1, batch_size)
        multi_hot_ids_l = []
        for k, (sparse_data_batch_for_table, multi_hot_table) in enumerate(
            zip(lS_i, self.multi_hot_tables_l)
        ):
            multi_hot_ids = torch.nn.functional.embedding(
                sparse_data_batch_for_table, multi_hot_table
            )
            multi_hot_ids = multi_hot_ids.reshape(-1)
            multi_hot_ids_l.append(multi_hot_ids)
            if self.collect_freqs_stats and (
                self.model_to_track is None or self.model_to_track.training
            ):
                idx_pre, cnt_pre = np.unique(sparse_data_batch_for_table, return_counts=True)
                idx_post, cnt_post = np.unique(multi_hot_ids, return_counts=True)
                self.freqs_pre_hash[k][idx_pre] += cnt_pre
                self.freqs_post_hash[k][idx_post] += cnt_post
        lS_i = torch.cat(multi_hot_ids_l)
        if batch_size == self.batch_size:
            return lS_i, self.offsets
        else:
            return lS_i, self.__make_offsets(
                self.multi_hot_sizes, self.num_embeddings_per_feature, batch_size
            )

    def convert_to_multi_hot(self, batch: Batch) -> Batch:
        batch_size = len(batch.dense_features)
        lS_i = batch.sparse_features._values
        lS_i, lS_o = self.__make_new_batch(lS_i, batch_size)
        new_sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=batch.sparse_features._keys,
            values=lS_i,
            offsets=lS_o,
        )
        return Batch(
            dense_features=batch.dense_features,
            sparse_features=new_sparse_features,
            labels=batch.labels,
        )
