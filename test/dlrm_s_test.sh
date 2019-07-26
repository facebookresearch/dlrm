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

dlrm_py="python dlrm_s_pytorch.py"
dlrm_c2="python dlrm_s_caffe2.py"

echo "Running commands ..."
#run pytorch
echo $dlrm_py
$dlrm_py --mini-batch-size=1 --data-size=1 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp1
$dlrm_py --mini-batch-size=2 --data-size=4 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp2
$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp3
$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=3 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp4

#run caffe2
echo $dlrm_c2
$dlrm_c2 --mini-batch-size=1 --data-size=1 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ccc1
$dlrm_c2 --mini-batch-size=2 --data-size=4 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ccc2
$dlrm_c2 --mini-batch-size=2 --data-size=5 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ccc3
$dlrm_c2 --mini-batch-size=2 --data-size=5 --nepochs=3 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ccc4

echo "Checking results ..."
#check results
#WARNING: correct test will have no difference in numeric values
#(but might have some verbal difference, e.g. due to warnnings)
#in the output file
echo "diff test1 (no numeric values in the output = SUCCESS)"
diff ccc1 ppp1
echo "diff test2 (no numeric values in the output = SUCCESS)"
diff ccc2 ppp2
echo "diff test3 (no numeric values in the output = SUCCESS)"
diff ccc3 ppp3
echo "diff test4 (no numeric values in the output = SUCCESS)"
diff ccc4 ppp4
