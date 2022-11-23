#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchx.specs as specs
from torchx.components.dist import ddp
from torchx.specs.api import Resource


def run_dlrm_main(num_trainers: int = 8, *script_args: str) -> specs.AppDef:
    """
    Args:
        num_trainers: The number of trainers to use.
        script_args: A variable number of parameters to provide dlrm_main.py.
    """
    cwd = os.getcwd()
    entrypoint = os.path.join(cwd, "dlrm_main.py")

    user = os.environ.get("USER")
    image = f"/data/home/{user}"

    if num_trainers > 8 and num_trainers % 8 != 0:
        raise ValueError(
            "Trainer jobs spanning multiple hosts must be in multiples of 8."
        )
    nproc_per_node = 8 if num_trainers >= 8 else num_trainers
    num_replicas = max(num_trainers // 8, 1)

    return ddp(
        *script_args,
        name="train_dlrm",
        image=image,
        # AWS p4d instance (https://aws.amazon.com/ec2/instance-types/p4/).
        cpu=96,
        gpu=8,
        memMB=-1,
        script=entrypoint,
        j=f"{num_replicas}x{nproc_per_node}",
    )
