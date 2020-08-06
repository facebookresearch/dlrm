#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch

dlrm_pt_bin="mpirun -n 2 python dlrm_s_pytorch.py"

# Choose the algorithm to run
if [[ $# == 1 ]]; then
    alg=$1
else
    alg=local_sgd
fi

if [[ $alg == post_local_sgd ]]; then
    alg_option="--distributed-optimization --alg=post_local_sgd --local-steps=2 --initial-steps=60 --initial-step-method=single_process"
elif [[ $alg == hierarchical_local_sgd ]]; then
    alg_option="--distributed-optimization --alg=hierarchical_local_sgd --num-nodes=1 --nprocs-per-node=2 --local-sync-freq=4 --global-sync-freq=4"
elif [[ $alg == noise_injection ]]; then
    alg_option="--distributed-optimization --alg=local_sgd --local-steps=1 --add-noise --noise-type gaussian --variance 0.001 --linear-variance-decay 0.01"
elif [[ $alg == slow_momentum ]]; then
    alg_option="--distributed-optimization --alg=local_sgd --local-steps=1 --slow-momentum --slow-momentum-factor=0.8 --slow-learning-rate=1.0 --inner-loop-steps 2"	
else
    # default algorithm is local SGD
    alg_option="--distributed-optimization --alg=local_sgd --local-steps=2"
fi

# WARNING: the following parameters will be set based on the data set
raw_data_file=~/datasets/criteo/dac/train.txt
processed_data_file=~/datasets/criteo/dac/kaggleAdDisplayChallenge_processed.npz

echo "run ${alg} on Criteo kaggle with two trainers using PyTorch ..."
echo "algorithm options = ${alg_option}"
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin --use-gpu --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=$raw_data_file --processed-data-file=$processed_data_file --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=16384 --print-freq=32 --print-time --test-mini-batch-size=256 --test-num-workers=16 --test-freq=2398 $alg_option 2>&1 | tee run_kaggle_pt.log

echo "done"
