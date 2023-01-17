#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from enum import Enum
from pprint import pprint
from typing import List, Optional, Union

import mlperf_logging.mllog as mllog
import mlperf_logging.mllog.constants as mllog_constants
import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from tqdm import tqdm

import mlperf_logging_utils

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:lr_scheduler
    from lr_scheduler import LRPolicyScheduler

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot, RestartableMap
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import get_dataloader  # noqa F811
    from .lr_scheduler import LRPolicyScheduler  # noqa F811
    from .multi_hot import Multihot, RestartableMap  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.

mllogger = mllog.get_mllogger()


class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.add_argument(
        "--validation_auroc",
        type=float,
        default=None,
        help="Validation AUROC threshold to stop training once reached.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument(
        "--print_progress",
        action="store_true",
        help="Print tqdm progress bar during training and evaluation.",
    )
    return parser.parse_args(argv)


def _evaluate(
    limit_batches: Optional[int],
    eval_pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str,
    epoch_num: Union[int, float],
    log_eval_samples: bool,
    disable_tqdm: bool,
) -> float:
    """
    Evaluates model. Computes and prints AUROC. Helper function for train_and_evaluate.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        eval_pipeline (TrainPipelineSparseDist): pipelined model.
        eval_dataloader (DataLoader): Dataloader for a validation set.
        stage (str): name of the stage (for logging purposes).
        epoch_num (int or float): Iterations passed as epoch fraction (for logging purposes).
        log_eval_samples (bool): Whether to print mllog with the number of samples
        disable_tqdm (bool): Whether to print tqdm progress bar.

    Returns:
        float: auroc result
    """
    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=disable_tqdm,
        )
        mllogger.start(
            key=mllog_constants.EVAL_START,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )

    eval_pipeline._model.eval()
    device = eval_pipeline._device

    # Set eval_pipeline._connected to False to cause the pipeline to refill with new batches as if it were newly created and empty.
    eval_pipeline._connected = False

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)
    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.
    two_filler_batches = itertools.islice(
        iter(eval_dataloader), TRAIN_PIPELINE_STAGES - 1
    )
    iterator = itertools.chain(iterator, two_filler_batches)

    auroc = metrics.AUROC(compute_on_step=False, task='binary').to(device)

    with torch.no_grad():
        try:
            while True:
                _loss, logits, labels = eval_pipeline.progress(iterator)
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                if is_rank_zero:
                    pbar.update(1)
        except StopIteration:
            # Dataset traversal complete
            pass

    auroc_result = auroc.compute().item()
    num_samples = torch.tensor(sum(map(len, auroc.target)), device=device)
    dist.reduce(num_samples, 0, op=dist.ReduceOp.SUM)
    num_samples = num_samples.item()

    if is_rank_zero:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Number of {stage} samples: {num_samples}")
        mllogger.event(
            key=mllog_constants.EVAL_ACCURACY,
            value=auroc_result,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
        mllogger.end(
            key=mllog_constants.EVAL_STOP,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
        if log_eval_samples:
            mllogger.event(
                key=mllog_constants.EVAL_SAMPLES,
                value=num_samples,
                metadata={mllog_constants.EPOCH_NUM: epoch_num},
            )
    return auroc_result


def _train(
    train_pipeline: TrainPipelineSparseDist,
    val_pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    lr_scheduler,
    print_lr: bool,
    validation_freq: Optional[int],
    validation_auroc: Optional[float],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int],
    disable_tqdm: bool,
) -> bool:
    """
    Trains model for 1 epoch. Helper function for train_and_evaluate.

    Args:
        train_pipeline (TrainPipelineSparseDist): pipelined model used for training.
        val_pipeline (TrainPipelineSparseDist): pipelined model used for validation.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        validation_auroc (Optional[float]): AUROC level desired for stopping training.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.
        disable_tqdm (bool): Whether to print tqdm progress bar.

    Returns:
        bool: Whether the validation_auroc threshold is reached.
    """
    train_pipeline._model.train()

    # Set train_pipeline._connected to False to cause the pipeline to refill with new batches as if it were newly created and empty.
    train_pipeline._connected = False

    iterator = itertools.islice(iter(train_dataloader), limit_train_batches)
    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.
    two_filler_batches = itertools.islice(
        iter(train_dataloader), TRAIN_PIPELINE_STAGES - 1
    )
    iterator = itertools.chain(iterator, two_filler_batches)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=disable_tqdm,
        )
    it = 1
    is_first_eval = True
    is_success = False
    try:
        for it in itertools.count(1):
            if is_rank_zero and print_lr:
                for i, g in enumerate(train_pipeline._optimizer.param_groups):
                    print(f"lr: {it} {i} {g['lr']:.6f}")
            train_pipeline.progress(iterator)
            lr_scheduler.step()
            if is_rank_zero:
                pbar.update(1)
            if validation_freq and it % validation_freq == 0:
                epoch_num = epoch + it / len(train_dataloader)
                auroc_result = _evaluate(
                    limit_val_batches,
                    val_pipeline,
                    val_dataloader,
                    "val",
                    epoch_num,
                    is_first_eval and epoch == 0,
                    disable_tqdm,
                )
                is_first_eval = False
                if validation_auroc is not None and auroc_result >= validation_auroc:
                    dist.barrier()
                    if is_rank_zero:
                        mllogger.end(
                            key=mllog_constants.RUN_STOP,
                            metadata={
                                mllog_constants.STATUS: mllog_constants.SUCCESS,
                                mllog_constants.EPOCH_NUM: epoch_num,
                            },
                        )
                    is_success = True
                    break
                train_pipeline._model.train()
    except StopIteration:
        # Dataset traversal complete
        pass

    if is_rank_zero:
        print("Total number of iterations:", it - 1)

    return is_success


def train_and_evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    lr_scheduler: LRPolicyScheduler,
) -> None:
    """
    Train/validation loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        None.
    """
    train_pipeline = TrainPipelineSparseDist(model, optimizer, device)
    val_pipeline = TrainPipelineSparseDist(model, optimizer, device)

    is_rank_zero = dist.get_rank() == 0
    is_success = False
    epoch = 0

    for epoch in range(args.epochs):
        if is_rank_zero:
            mllogger.start(
                key=mllog_constants.EPOCH_START,
                metadata={mllog_constants.EPOCH_NUM: epoch},
            )
        is_success = _train(
            train_pipeline,
            val_pipeline,
            train_dataloader,
            val_dataloader,
            epoch,
            lr_scheduler,
            args.print_lr,
            args.validation_freq_within_epoch,
            args.validation_auroc,
            args.limit_train_batches,
            args.limit_val_batches,
            not args.print_progress,
        )
        if is_rank_zero:
            mllogger.end(
                key=mllog_constants.EPOCH_STOP,
                metadata={mllog_constants.EPOCH_NUM: epoch},
            )
        if is_success:
            break

    dist.barrier()
    if not is_success and is_rank_zero:
        # Run status "aborted" is reported in the case AUROC threshold is not met
        mllogger.end(
            key=mllog_constants.RUN_STOP,
            metadata={
                mllog_constants.STATUS: mllog_constants.ABORTED,
                mllog_constants.EPOCH_NUM: epoch + 1,
            },
        )


def main(argv: List[str]) -> None:
    """
    Trains and validates a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    # The reference implementation does not clear the cache currently
    # but the submissions are required to do so
    mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True)
    mllogger.start(key=mllog_constants.INIT_START)

    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    if args.multi_hot_sizes is not None:
        assert (
            args.num_embeddings_per_feature is not None
            and len(args.multi_hot_sizes) == len(args.num_embeddings_per_feature)
            or args.num_embeddings_per_feature is None
            and len(args.multi_hot_sizes) == len(DEFAULT_CAT_NAMES)
        ), "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
    assert (
        args.in_memory_binary_criteo_path is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--in_memory_binary_criteo_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
    assert (
        args.multi_hot_sizes is None or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_sizes is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    assert (
        args.multi_hot_distribution_type is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pprint(vars(args))
        mlperf_logging_utils.info(mllogger, "dcnv2", "reference_implementation")
        mllogger.event(
            key=mllog_constants.GLOBAL_BATCH_SIZE,
            value=dist.get_world_size() * args.batch_size,
        )
        mllogger.event(
            key=mllog_constants.OPT_BASE_LR,
            value=args.learning_rate,
        )
        mllogger.event(
            key=mllog_constants.GRADIENT_ACCUMULATION_STEPS,
            value=1,  # Gradient accumulation is not supported in the reference implementation
        )
        mllogger.event(
            key=mllog_constants.SEED,
            value=args.seed,  # TODO: seed has to be initialized properly and synced between devices
        )

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.in_memory_binary_criteo_path is None
        and args.synthetic_multi_hot_criteo_path is None
    ):
        for split in ["train", "val"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            if args.num_embeddings is None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    if args.interaction_type == InteractionType.ORIGINAL:
        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.DCN:
        dlrm_model = DLRM_DCN(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dcn_num_layers=args.dcn_num_layers,
            dcn_low_rank_dim=args.dcn_low_rank_dim,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.PROJECTION:
        dlrm_model = DLRM_Projection(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
            interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
            dense_device=device,
        )
    else:
        raise ValueError(
            "Unknown interaction option set. Should be original, dcn, or projection."
        )

    train_model = DLRMTrain(dlrm_model)
    embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
    # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
    # the optimizer update will be applied in the backward pass, in this case through a fused op.
    # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
    # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678
    apply_optimizer_in_backward(
        embedding_optimizer,
        train_model.model.sparse_arch.parameters(),
        {"lr": args.learning_rate},
    )
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If experience OOM, increase the percentage. see
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(
        train_model, get_default_sharders(), dist.GroupMember.WORLD
    )

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )
    if is_rank_zero and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate)
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    lr_scheduler = LRPolicyScheduler(
        optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
    )

    dist.barrier()
    if is_rank_zero:
        mllogger.start(key=mllog_constants.INIT_STOP)
    dist.barrier()
    if is_rank_zero:
        mllogger.start(key=mllog_constants.RUN_START)
    dist.barrier()

    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")

    if is_rank_zero:
        mllogger.event(
            key=mllog_constants.TRAIN_SAMPLES,
            value=dist.get_world_size() * len(train_dataloader) * args.batch_size,
        )

    if args.multi_hot_sizes is not None:
        multihot = Multihot(
            args.multi_hot_sizes,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats=args.collect_multi_hot_freqs_stats,
            dist_type=args.multi_hot_distribution_type,
        )
        multihot.pause_stats_collection_during_val(model)
        train_dataloader = RestartableMap(
            multihot.convert_to_multi_hot, train_dataloader
        )
        val_dataloader = RestartableMap(multihot.convert_to_multi_hot, val_dataloader)
    train_and_evaluate(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )
    if args.collect_multi_hot_freqs_stats:
        multihot.save_freqs_stats()


if __name__ == "__main__":
    main(sys.argv[1:])
