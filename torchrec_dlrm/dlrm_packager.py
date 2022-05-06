#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from typing import List

from modules.dlrm_predict import (
    DLRMPredictFactory,
    DLRMModelConfig,
)
from torch.package import PackageExporter
from torchrec.datasets.criteo import DEFAULT_INT_NAMES, DEFAULT_CAT_NAMES
from torchrec.inference.model_packager import PredictFactoryPackager

# OSS Only


class DLRMPredictFactoryPackager(PredictFactoryPackager):
    @classmethod
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        pe.extern(
            [
                "io",
                "_imp",
                "_ctypes",
                "_string",
                "numpy.**",
                "pandas.**",
                "pyarrow.**",
                "six.moves",
                "sys",
            ]
        )

    @classmethod
    def set_mocked_modules(cls, pe: PackageExporter) -> None:
        pe.mock(
            [
                "markupsafe._speedups",
            ]
        )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm model packager")
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,"
        "10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--sparse_feature_names",
        type=str,
        default=",".join(DEFAULT_CAT_NAMES),
        help="Comma separated names of the sparse features.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--num_dense_features",
        type=int,
        default=len(DEFAULT_INT_NAMES),
        help="Number of dense features.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path of model package.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    Use torch.package to package the torchrec DLRM Model.

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)

    model_config = DLRMModelConfig(
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        dense_in_features=args.num_dense_features,
        embedding_dim=args.embedding_dim,
        id_list_features_keys=args.sparse_feature_names.split(","),
        num_embeddings_per_feature=list(
            map(int, args.num_embeddings_per_feature.split(","))
        ),
        num_embeddings=args.num_embeddings,
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
    )

    DLRMPredictFactoryPackager.save_predict_factory(
        DLRMPredictFactory, {"config.pkl": model_config}, args.output_path
    )
    print(f"Package is saved to {args.output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
