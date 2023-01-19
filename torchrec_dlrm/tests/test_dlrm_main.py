#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
import uuid

from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec import test_utils
from torchrec.datasets.test_utils.criteo_test_utils import CriteoTest

from ..dlrm_main import main


class MainTest(unittest.TestCase):
    @classmethod
    def _run_trainer_random(cls) -> None:
        main(
            [
                "--limit_train_batches",
                "10",
                "--limit_val_batches",
                "8",
                "--limit_test_batches",
                "6",
                "--over_arch_layer_sizes",
                "8,1",
                "--dense_arch_layer_sizes",
                "8,8",
                "--embedding_dim",
                "8",
                "--num_embeddings",
                "8",
            ]
        )

    @test_utils.skip_if_asan
    def test_main_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer_random)()

    @classmethod
    def _run_trainer_criteo_in_memory(cls) -> None:
        with CriteoTest._create_dataset_npys(
            num_rows=50, filenames=[f"day_{i}" for i in range(24)]
        ) as files:
            main(
                [
                    "--over_arch_layer_sizes",
                    "8,1",
                    "--dense_arch_layer_sizes",
                    "8,8",
                    "--embedding_dim",
                    "8",
                    "--num_embeddings",
                    "64",
                    "--batch_size",
                    "2",
                    "--in_memory_binary_criteo_path",
                    os.path.dirname(files[0]),
                    "--epochs",
                    "2",
                ]
            )

    @test_utils.skip_if_asan
    def test_main_function_criteo_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer_criteo_in_memory)()

    @classmethod
    def _run_trainer_dcn(cls) -> None:
        with CriteoTest._create_dataset_npys(
            num_rows=50, filenames=[f"day_{i}" for i in range(24)]
        ) as files:
            main(
                [
                    "--over_arch_layer_sizes",
                    "8,1",
                    "--dense_arch_layer_sizes",
                    "8,8",
                    "--embedding_dim",
                    "8",
                    "--num_embeddings",
                    "64",
                    "--batch_size",
                    "2",
                    "--in_memory_binary_criteo_path",
                    os.path.dirname(files[0]),
                    "--epochs",
                    "2",
                    "--interaction_type",
                    "dcn",
                    "--dcn_num_layers",
                    "2",
                    "--dcn_low_rank_dim",
                    "8",
                ]
            )

    @test_utils.skip_if_asan
    def test_main_function_dcn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer_dcn)()
