# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: run dataset pre-processing in standalone mode
# WARNING: These steps are required to work with Cython
# 1. Instal Cython
# > sudo yum install Cython
# 2. Please copy data_utils.py into data_utils_cython.pyx
# 3. Compile the data_utils_cython.pyx to generate .so
# (it's important to keep extension .pyx rather than .py
#  to ensure the C/C++ .so no .py is loaded at import time)
# > python cython_compile.py build_ext --inplace
# This should create data_utils_cython.so, which can be loaded below with "import"
# 4. Run standalone datatset preprocessing to generate .npz files
# a. Kaggle
# > python cython_criteo.py --data-set=kaggle --raw-data-file=./input/train.txt
#   --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz
# b. Terabyte
# > python cython_criteo.py --max-ind-range=10000000 [--memory-map] --data-set=terabyte
#   --raw-data-file=./input/day --processed-data-file=./input/terabyte_processed.npz

from __future__ import absolute_import, division, print_function, unicode_literals

import data_utils_cython as duc

if __name__ == "__main__":
    ### import packages ###
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Preprocess Criteo dataset"
    )
    # model related parameters
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    args = parser.parse_args()

    duc.loadDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map
    )
