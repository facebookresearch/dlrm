#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2
#WARNING: must have openmpi to run this test

echo "Running tests for distributed optimization implementations"
echo "The tests require the number of CUDA devices no less than two"

dlrm_py="mpirun -n 2 python dlrm_s_pytorch.py"

echo "test mini-batch SGD"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 2>&1 | tee test_mini_batch_sgd.log

echo "test local SGD"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 2>&1 | tee test_local_sgd.log

echo "test post-local SGD version single_process"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 --initial-steps 50 --initial-step-method single_process 2>&1 | tee test_version_single_post_local_sgd.log

echo "test post-local SGD version multiple_processes"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 --initial-steps 50 --initial-step-method multiple_processes 2>&1 | tee test_version_multi_post_local_sgd_2.log

echo "test slow momentum"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 --slow-momentum --slow-momentum-factor 0.9 --slow-learning-rate 1.0 --inner-loop-steps 4 2>&1 | tee test_slow_momentum.log

echo "test Gaussian noise injection"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 --add-noise --noise-type gaussian --variance 0.0001 --linear-variance-decay 0.01 2>&1 | tee test_gaussian_noise_injection.log

echo "test multiplicative Gaussian noise injection"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg local_sgd --local-steps 2 --add-noise --noise-type multiplicative_gaussian --variance 0.0001 --linear-variance-decay 0.01 2>&1 | tee test_multiplicative_gaussian_noise_injection.log

echo "test hierarchical local SGD"
$dlrm_py --use-gpu --num-batches 100 --mini-batch-size 256 --distributed-optimization --alg hierarchical_local_sgd --num-nodes 1 --nprocs-per-node 2 --local-sync-freq 4 --global-sync-freq 4 2>&1 | tee test_local_sgd.log

echo "tests finished"
