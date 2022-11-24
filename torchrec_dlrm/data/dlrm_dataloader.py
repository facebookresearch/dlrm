#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DAYS,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset

STAGES = ["train", "val", "test"]


def _get_random_dataloader(
    args: argparse.Namespace,
) -> DataLoader:
    return DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=args.batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature
            if hasattr(args, "num_embeddings_per_feature")
            else None,
            manual_seed=args.seed if hasattr(args, "seed") else None,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=args.pin_memory,
        num_workers=0,
    )


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    dir_name = args.in_memory_binary_criteo_path
    if stage == "train":
        stage_files: List[List[str]] = [
            [os.path.join(dir_name, f"day_{i}_dense.npy") for i in range(DAYS-1)],
            [os.path.join(dir_name, f"day_{i}_sparse.npy") for i in range(DAYS-1)],
            [os.path.join(dir_name, f"day_{i}_labels.npy") for i in range(DAYS-1)],
        ]
    elif stage in ["val", "test"]:
        stage_files: List[List[str]] = [
            [os.path.join(dir_name, f"day_{DAYS-1}_dense.npy")],
            [os.path.join(dir_name, f"day_{DAYS-1}_sparse.npy")],
            [os.path.join(dir_name, f"day_{DAYS-1}_labels.npy")],
        ]
    if stage in ["val", "test"] and args.test_batch_size is not None:
        batch_size = args.test_batch_size
    else:
        batch_size =  args.batch_size
    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            stage,
            *stage_files,  # pyre-ignore[6]
            batch_size=batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            shuffle_batches=args.shuffle_batches,
            shuffle_training_set=args.shuffle_training_set,
            shuffle_training_set_random_seed=args.seed,
            mmap_mode=args.mmap_mode,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (
        (backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory
    )

    if (
        not hasattr(args, "in_memory_binary_criteo_path")
        or args.in_memory_binary_criteo_path is None
    ):
        return _get_random_dataloader(args)
    else:
        return _get_in_memory_dataloader(args, stage)
