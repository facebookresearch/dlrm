from typing import (
    List,
    Tuple,
)

import torch
import numpy as np
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class RestartableMap:
    def __init__(self, f, source):
        self.source = source
        self.func = f

    def __iter__(self):
        for x in self.source:
            yield self.func(x)

class Multihot():
    def __init__(
        self,
        multi_hot_size: int,
        multi_hot_min_table_size: int,
        ln_emb: List[int],
        batch_size: int,
        collect_freqs_stats: bool,
        dist_type: str = "uniform",
    ):
        if dist_type not in {"uniform", "pareto"}:
            raise ValueError(
                "Multi-hot distribution type {} is not supported."
                "Only \"uniform\" and \"pareto\" are supported.".format(dist_type)
            )
        self.dist_type = dist_type
        self.multi_hot_min_table_size = multi_hot_min_table_size
        self.multi_hot_size = multi_hot_size
        self.batch_size = batch_size
        self.ln_emb = ln_emb
        self.lS_i_offsets_cache = self.__make_multi_hot_indices_cache(multi_hot_size, ln_emb)
        self.lS_o_cache = self.__make_offsets_cache(multi_hot_size, multi_hot_min_table_size, ln_emb, batch_size)

        # For plotting frequency access
        self.collect_freqs_stats = collect_freqs_stats
        self.model_to_track = None
        self.freqs_pre_hash = []
        self.freqs_post_hash = []
        for row_count in ln_emb:
            self.freqs_pre_hash.append(np.zeros((row_count)))
            self.freqs_post_hash.append(np.zeros((row_count)))

    def save_freqs_stats(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        pre_dict = {str(k) : e for k, e in enumerate(self.freqs_pre_hash)}
        np.save(f"stats_pre_hash_{rank}_pareto.npy", pre_dict)
        post_dict = {str(k) : e for k, e in enumerate(self.freqs_post_hash)}
        np.save(f"stats_post_hash_{rank}_pareto.npy", post_dict)

    def pause_stats_collection_during_val_and_test(self, model: torch.nn.Module) -> None:
        self.model_to_track = model

    def __make_multi_hot_indices_cache(
        self,
        multi_hot_size: int,
        ln_emb: List[int],
    ) -> List[np.array]:
        cache = [ np.zeros((rows_count, multi_hot_size)) for rows_count in ln_emb ]
        for k, e in enumerate(ln_emb):
            np.random.seed(k) # The seed is necessary for all ranks to produce the same lookup values.
            if self.dist_type == "uniform":
                cache[k][:,1:] = np.random.randint(0, e, size=(e, multi_hot_size-1))
            elif self.dist_type == "pareto":
                cache[k][:,1:] = np.random.pareto(a=0.25, size=(e, multi_hot_size-1)).astype(np.int32) % e
        # cache axes are [table, batch, offset]
        cache = [ torch.from_numpy(table_cache).int() for table_cache in cache ]
        return cache

    def __make_offsets_cache(
        self,
        multi_hot_size: int,
        multi_hot_min_table_size: int,
        ln_emb: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        lS_o = torch.ones((len(ln_emb) * self.batch_size), dtype=torch.int32)
        for cf, table_length in enumerate(ln_emb):
            if table_length >= multi_hot_min_table_size:
                lS_o[cf*batch_size : (cf+1)*batch_size] = multi_hot_size
        lS_o = torch.cumsum( torch.concat((torch.tensor([0]), lS_o)), axis=0)
        return lS_o

    def __make_new_batch(
        self,
        lS_o: torch.Tensor,
        lS_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lS_i = lS_i.reshape(-1, self.batch_size)
        if 1 < self.multi_hot_size:
            multi_hot_i_l = []
            for cf, table_length in enumerate(self.ln_emb):
                if table_length < self.multi_hot_min_table_size:
                    multi_hot_i_l.append(lS_i[cf])
                else:
                    keys = lS_i[cf]
                    multi_hot_i = torch.nn.functional.embedding(keys, self.lS_i_offsets_cache[cf])
                    multi_hot_i[:,0] = keys
                    multi_hot_i = multi_hot_i.reshape(-1)
                    multi_hot_i_l.append(multi_hot_i)
                    if self.collect_freqs_stats and (
                        self.model_to_track is None or self.model_to_track.training
                    ):
                        self.freqs_pre_hash[cf][lS_i[cf]] += 1
                        self.freqs_post_hash[cf][multi_hot_i] += 1
            lS_i = torch.cat(multi_hot_i_l)
            return self.lS_o_cache, lS_i
        else:
            lS_i = torch.cat(lS_i)
            return lS_o, lS_i

    def convert_to_multi_hot(self, batch: Batch) -> Batch:
        lS_i = batch.sparse_features._values
        lS_o = batch.sparse_features._offsets
        lS_o, lS_i = self.__make_new_batch(lS_o, lS_i)
        new_sparse_features=KeyedJaggedTensor.from_offsets_sync(
            keys=batch.sparse_features._keys,
            values=lS_i,
            offsets=lS_o,
        )
        return Batch(
            dense_features=batch.dense_features,
            sparse_features=new_sparse_features,
            labels=batch.labels,
        )
