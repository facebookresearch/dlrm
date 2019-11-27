#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

python dlrm_s_pytorch.py --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" \
                         --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 --data-generation=dataset \
                         --data-set=terabyte --raw-data-file=/data/day --memory-map --use-gpu \
                         --processed-data-file=/data/terabyte_processed.npz --loss-function=bce --round-targets=True \
                          --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1024 --print-time \
                          --test-mini-batch-size=16384 --test-num-workers=16
