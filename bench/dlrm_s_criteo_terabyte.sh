#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 --data-generation=dataset --data-set=terabyte --raw-data-file=./input/day --processed-data-file=./input/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "run caffe2 ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_c2_bin --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 --data-generation=dataset --data-set=terabyte --raw-data-file=./input/day --processed-data-file=./input/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_terabyte_c2.log

echo "done"
