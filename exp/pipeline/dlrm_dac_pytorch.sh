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

dlrm_pt_bin="python main_with_runtime.py"
datagen=dataset
print_freq=256 # 256 # fix
nepochs=1

mini_batch_size=64
num_batches=2560 #613937 # fix
num_workers=4

test_num_batches=200 # fix
test_mini_batch_size=16384 #fix
test_num_workers=16

num_input_rank=3
nrank=6
ngpu=$((nrank-1))
conf_file=hybrid_conf.json
exp_name=tmp

#--mini-batch-size=64 --print-freq=256 --test-freq=1024 --print-time --test-mini-batch-size=128 --test-num-workers=16
#--arch-embedding-size="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"\

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
for r in $(seq 0 $num_input_rank); do
    echo "runing input rank $r"
    $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"\
        --arch-embedding-size="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"\
        --data-generation=$datagen --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz\
        --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=$mini_batch_size\
        --print-freq=$print_freq --print-time --test-mini-batch-size=$test_mini_batch_size --test-num-workers=$test_num_workers\
        --module models.dlrm.gpus=3 --rank $r --local-rank $r --master-addr 127.0.0.1\
        --config-path models/dlrm/gpus\=3/$conf_file --distributed-backend gloo --num-ranks-in-server $nrank\
        --use-gpu --num-batches $num_batches --test-num-batches $test_num_batches --nepochs $nepochs --print-freq=$print_freq $dlrm_extra_option 2>&1 > $exp_name/run_pt_$r.log &
done

datagen=random
for r in $(seq $((num_input_rank+1)) $ngpu); do
    echo "runing rank $r"
    $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1"\
        --arch-embedding-size="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572"\
        --data-generation=$datagen --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz\
        --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=$mini_batch_size\
        --print-freq=$print_freq --print-time --test-mini-batch-size=$test_mini_batch_size --test-num-workers=$test_num_workers\
        --module models.dlrm.gpus=3 --rank $r --local-rank $r --master-addr 127.0.0.1\
        --config-path models/dlrm/gpus\=3/$conf_file --distributed-backend gloo --num-ranks-in-server $nrank\
        --use-gpu --num-batches $num_batches --test-num-batches $test_num_batches --nepochs $nepochs --print-freq=$print_freq $dlrm_extra_option 2>&1 > $exp_name/run_pt_$r.log &
done

echo "done"
