import sys
import argparse

import torch_xla.distributed.xla_multiprocessing as xmp

from dlrm_s_pytorch import main, parse_args


if __name__ == '__main__':
    pre_spawn_parser = argparse.ArgumentParser()
    pre_spawn_parser.add_argument(
        "--tpu-cores", type=int, default=8, choices=[1, 8]
    )
    pre_spawn_flags, _ = pre_spawn_parser.parse_known_args()
    xmp.spawn(main, args=(), nprocs=pre_spawn_flags.tpu_cores)
