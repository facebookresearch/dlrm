#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have fbgemm_gpu module to run these tests.

echo -e "\nConsistency test: fbgemm_gpu -compared-with- PyTorch emb ops"
dlrm_base_config_="python dlrm_s_pytorch.py --arch-sparse-feature-size=172 --arch-mlp-bot=1559-2500-2500-172 --arch-mlp-top=2000-2000-2000-1 --arch-embedding-size=213728-213728-213728-213728-213728-213728-213728-213728 --mini-batch-size=64 --num-indices-per-lookup-fixed=1 --num-indices-per-lookup=16 --num-batches=1 --nepochs=3 --debug-mode"

for weighted_pooling in '' ' --weighted-pooling=fixed' ' --weighted-pooling=learned';
do
    dlrm_base_config=$dlrm_base_config_$weighted_pooling

    echo -e "\n======================================================"
    echo "Testing 32-bit embeddings"

    dlrm_config="$dlrm_base_config"
    echo "---GROUND TRUTH--- using PyTorch emb ops on CPU"
    echo "$dlrm_config"
    $dlrm_config > aaa1
    echo "---COMPARISON--- using fbgemm_gpu on CPU"
    echo "$dlrm_config --use-fbgemm-gpu"
    $dlrm_config --use-fbgemm-gpu > aaa2
    echo "diff GT & COMP (no numeric values in the output = SUCCESS)"
    diff aaa1 aaa2

    echo "---GROUND TRUTH--- using PyTorch emb ops on GPU"
    echo "$dlrm_config --use-gpu"
    $dlrm_config --use-gpu > bbb1
    echo "---COMPARISON--- using fbgemm_gpu on GPU"
    echo "$dlrm_config --use-gpu --use-fbgemm-gpu"
    $dlrm_config --use-fbgemm-gpu --use-gpu > bbb2
    echo "diff GT & COMP (no numeric values in the output = SUCCESS)"
    diff bbb1 bbb2

    echo -e "\n======================================================"
    echo "Testing 8-bit quantized embeddings, inference only"
    dlrm_config="$dlrm_base_config --inference-only --quantize-emb-with-bit=8"

    echo "---GROUND TRUTH--- using PyTorch emb ops on CPU"
    echo "$dlrm_config"
    $dlrm_config > ccc1

    echo "---COMPARISON--- using fbgemm_gpu on CPU"
    echo "$dlrm_config --use-fbgemm-gpu"
    $dlrm_config --use-fbgemm-gpu > ccc2
    echo "diff GT & COMP (no numeric values in the output = SUCCESS)"
    diff ccc1 ccc2
done
