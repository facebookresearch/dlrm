#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import os
import shutil
import sys

import numpy as np
import torch
from torch import distributed as dist, nn
from torchrec.datasets.criteo import DAYS

p = pathlib.Path(__file__).absolute().parents[1].resolve()
sys.path.append(os.fspath(p))

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot
except ImportError:
    pass

# internal import
try:
    from .multi_hot import Multihot  # noqa F811
except ImportError:
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to materialize synthetic multi-hot dataset into NumPy npz file format."
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        required=True,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to outputted multi-hot sparse dataset.",
    )
    parser.add_argument(
        "--copy_labels_and_dense",
        dest="copy_labels_and_dense",
        action="store_true",
        help="Flag to determine whether to copy labels and dense data to the output directory.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        required=True,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        required=True,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default="uniform",
        help="Multi-hot distribution options.",
    )
    return parser.parse_args()


def main() -> None:
    """
    This script generates and saves the MLPerf v2 multi-hot dataset (4 TB in size).
    First, run process_Criteo_1TB_Click_Logs_dataset.sh.
    Then, run this script as follows:

        python materialize_synthetic_multihot_dataset.py \
            --in_memory_binary_criteo_path $PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH \
            --output_path $MATERIALIZED_DATASET_PATH \
            --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
            --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
            --multi_hot_distribution_type uniform

    This script takes about 2 hours to run (can be parallelized if needed).
    """
    args = parse_args()
    for name, val in vars(args).items():
        try: vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError): pass
    try:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except (KeyError, ValueError):
        rank = 0
        world_size = 1

    print("Generating one-hot to multi-hot lookup table.")
    multihot = Multihot(
        multi_hot_sizes=args.multi_hot_sizes,
        num_embeddings_per_feature=args.num_embeddings_per_feature,
        batch_size=1, # Doesn't matter
        collect_freqs_stats=False,
        dist_type=args.multi_hot_distribution_type,
    )

    os.makedirs(args.output_path, exist_ok=True)

    for i in range(rank, DAYS, world_size):
        input_file_path = os.path.join(args.in_memory_binary_criteo_path, f"day_{i}_sparse.npy")
        print(f"Materializing {input_file_path}")
        sparse_data = np.load(input_file_path, mmap_mode = 'r')
        multi_hot_ids_dict = {}
        for j, (multi_hot_table, hash) in enumerate(zip(multihot.multi_hot_tables_l, args.num_embeddings_per_feature)):
            sparse_tensor = torch.from_numpy(sparse_data[:,j] % hash)
            multi_hot_ids_dict[str(j)] = nn.functional.embedding(sparse_tensor, multi_hot_table).numpy()
        output_file_path = os.path.join(args.output_path, f"day_{i}_sparse_multi_hot.npz")
        np.savez(output_file_path, **multi_hot_ids_dict)
        if args.copy_labels_and_dense:
            for part in ["labels", "dense"]:
                source_path = os.path.join(args.in_memory_binary_criteo_path, f"day_{i}_{part}.npy")
                output_path = os.path.join(args.output_path, f"day_{i}_{part}.npy")
                shutil.copyfile(source_path, output_path)
                print(f"Copying {source_path} to {output_path}")

if __name__ == "__main__":
    main()
