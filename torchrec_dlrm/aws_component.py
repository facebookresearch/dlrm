#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchx.specs as specs
from torchx.components.base import torch_dist_role
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

    return specs.AppDef(
        name="train_dlrm",
        roles=[
            torch_dist_role(
                name="trainer",
                image=image,
                # AWS p4d instance (https://aws.amazon.com/ec2/instance-types/p4/).
                resource=Resource(
                    cpu=96,
                    gpu=8,
                    memMB=-1,
                ),
                nnodes=num_replicas,
                entrypoint=entrypoint,
                nproc_per_node=nproc_per_node,
                rdzv_backend="c10d",
                args=script_args,
                rdzv_endpoint="localhost",
                rdzv_id="54321",
            ),
        ],
    )
