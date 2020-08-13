# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
import sys


# data generation
import data.dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print('Unable to import onnx. ', error)

# pytorch
import torch
# mixed-dimension trick
from tricks.md_embedding_bag import md_solver
# DLRM_Net model
from dlrm_s_pytorch import DLRM_Net
from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter


# pipeline runtime implementation
sys.path.append("..") #TODO: temporary
import runtime
import sgd
import importlib
from pipeline_utils import * #TODO: specify specific modules
from profiler.profile_utils import AverageMeter

exc = getattr(builtins, "IOError", "FileNotFoundError")


# compute BCE loss with shifted logits
# We use weighted BCE loss function because shifted logits
# only make sense for sampled data, and sampled data is
# supposed to have per-sample weights.
def shifted_binary_cross_entropy_pt(z, t, wts, lc, eps):
    # avoid inf
    zp, zp1 = torch.max(z, eps), torch.max(1 - z, eps)
    # compute shifter logits
    logits_shifted = torch.log(zp / zp1) + lc
    p_shifted = torch.sigmoid(logits_shifted)
    # compute BCE loss
    loss = nn.BCELoss(weight=wts.to(p_shifted.device), reduction="sum")
    E_shifted = loss(p_shifted, t.to(p_shifted.device))

    # return loss and shifted probabilities
    return E_shifted, p_shifted.detach().cpu().numpy()


if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-batches", type=int, default=0)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)

    # pipeline runtime implementation
    parser.add_argument('--config-path', default=None, type=str,
                    help="Path of configuration file")
    parser.add_argument('--distributed-backend', type=str,
                        help='distributed backend to use (gloo|nccl)')
    parser.add_argument('--no-input-pipelining', action='store_true',
                        help="No pipelining of inputs")
    parser.add_argument('--rank', default=None, type=int,
                        help="Rank of worker")
    parser.add_argument('--local-rank', default=0, type=int,
                        help="Local rank of worker (used as GPU id)")
    parser.add_argument('--num-ranks-in-server', default=1, type=int,
                    help="number of gpus per machine")
    parser.add_argument('--master-addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
    parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
    parser.add_argument('-v', '--verbose-frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")


    # pipeline optimization
    # Recompute tensors from forward pass, instead of saving them.
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute tensors in backward pass')
    # Macrobatching reduces the number of weight versions to save,
    # by not applying updates every minibatch.
    parser.add_argument('--macrobatch', action='store_true',
                        help='Macrobatch updates to save memory')
    parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
    parser.add_argument('--loss-scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')



    args = parser.parse_args()
    print("finish processing args")


    # Pipeline helper methods.
    def is_first_stage():
        return args.stage is None or (args.stage == 0)

    def is_last_stage():
        return args.stage is None or (args.stage == (args.num_stages-1))

    # pipeline configuration
    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)


    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        # pipeline implementation
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        ngpus = torch.cuda.device_count()  # 1
        print("Total {} GPU(s), using GPU {}...".format(ngpus, args.local_rank))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = \
            dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader_pipeline(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = args.test_num_batches if args.test_num_batches > 0 else len(test_ld)

    print("train data nbatches: ", nbatches, " test data nbatches: ", nbatches_test)

    def unpack_batch(b):
        # Experiment with unweighted samples
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size())

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W = unpack_batch(inputBatch)

            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    # pipeline implementation import the generated module
    module = importlib.import_module(args.module)

    # pipeline: move the loss function definition to model definition
    loss_ws = None # placeholder for LossFnWrapper
    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    # pipeline: use the loss function wrapper
    criterion = LossFnWrapper(loss_fn, args.use_gpu, device, args, loss_ws)
    model = module.model(criterion)

    # create stages for the model
    training_tensor_shapes = dict()
    dtypes = dict()
    inputs_module_destinations = dict()

    # obtain one batch for setting up the
    #print("train_ld len: ", len(train_ld))
    for j, inputBatch in enumerate(train_ld):
        X, lS_o, lS_i, T, W = unpack_batch(inputBatch)
        # print(X.shape)
        # print(lS_o)
        # print(lS_i)
        # print(len(T))
        # import sys; sys.stdout.flush()
        if j >= 0:
            break

    # pipeline: debug
    #print("X: ", X, X.size(), X.dtype)
    # input0: dense features
    training_tensor_shapes["input0"] = X.size()
    dtypes["input0"] = X.dtype
    inputs_module_destinations["input0"] = 0

    # if isinstance(lS_i, list):
        # pipeline: debug
        #print("lS_i: ", lS_i, len(lS_i), lS_i[0].size(), lS_i[0].dtype)
        #print("lS_o: ", lS_o, len(lS_o))
        # index, offset for input i, input i+1
    assert(len(lS_o) == len(lS_i))
    for k, S_i in enumerate(lS_i):
        S_o = lS_o[k]
        training_tensor_shapes["input"+str(2*k+1)] = S_i.size()
        dtypes["input"+str(2*k+1)] = S_i.dtype
        inputs_module_destinations["input"+str(2*k+1)] = 0

        training_tensor_shapes["input"+str(2*k+2)] = S_o.size()
        dtypes["input"+str(2*k+2)] = S_o.dtype
        inputs_module_destinations["input"+str(2*k+2)] = 0
    # else:
    #     raise Exception("Not Implement Yet.")

    # setting up the target tensor
    training_tensor_shapes["target"] = T.size()
    training_tensor_shapes["target_length"] = [args.mini_batch_size]
    dtypes["target"] = torch.float32 # T.dtype
    target_tensor_names = {"target"}
    # pipeline: debug
    #print(model[-1])
    #print(training_tensor_shapes, dtypes, inputs_module_destinations)
    for module_id, (stage, inputs, outputs) in enumerate(model[:-1]): # skip the last loss layer
        input_tensors = list()
        #print(inputs)
        for module_input in inputs:
            #print(module_input)
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id
            #if "input" in module_input: #TODO: further check.
            input_tensor = torch.ones(training_tensor_shapes[module_input],
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage.cuda()

        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs, list(output_tensors)):
            #output_tensor.cuda()
            training_tensor_shapes[output] = output_tensor.size()
            dtypes[output] = output_tensor.dtype
        # if configuration_maps['module_to_stage_map'] != None and\
        #     configuration_maps['stage_to_rank_map'] != None:
        #     stage_id = configuration_maps['module_to_stage_map'][module_id]
        #     rank_ids = configuration_maps['stage_to_rank_map'][stage_id]
        #     if args.local_rank in rank_ids:
        #         stage.cuda()

    eval_tensor_shapes = dict()
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [args.test_mini_batch_size] + list(training_tensor_shapes[key][1:])
        )
        training_tensor_shapes[key] = tuple(list(training_tensor_shapes[key]))

    #print("entering sampler setup")
    #import sys; sys.stdout.flush()
    # setup sampler
    distributed_sampler = False
    num_ranks_in_first_stage = 1
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
    if num_ranks_in_first_stage > 1:
        distributed_sampler = True
        # train_sampler for set epoch (for correct data shuffling)
        if args.data_generation == "dataset":
            train_ld, test_ld, train_sampler = \
                dp.make_criteo_loaders_with_sampler(args, train_data, test_data, num_ranks_in_first_stage)
        else:
            train_ld, test_ld, train_sampler = \
                dp.make_random_loader_with_sampler(args, train_data, test_data, num_ranks_in_first_stage)
    # print("finish sampler setup")
    # import sys; sys.stdout.flush()

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.RECOMMENDATION,
        enable_recompute=args.recompute)
    print("init runtime", r)
    import sys; sys.stdout.flush()
    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # TODO: support other optimizers
    optimizer = sgd.SGDWithWeightStashing(
            modules=r.modules(), master_parameters=r.master_parameters,
            model_parameters=r.model_parameters, loss_scale=args.loss_scale,
            num_versions=num_versions, lr=args.learning_rate)

    # # added
    # distributed_sampler = False
    # if configuration_maps['stage_to_rank_map'] is not None:
    #     num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
    #     if num_ranks_in_first_stage > 1:
    #         distributed_sampler = True

    def cal_accuracy(output, target):
        # output: model output
        # target: the corresponding target (T)
        with torch.no_grad():
            Z = output.clone()
            T = target.clone()
            lc = torch.log(torch.Tensor([1.0])).to(Z.device)  # 1.0 for the default bias
            eps = torch.Tensor([2.0**(-53)]).to(Z.device)

            (E_shifted, S_shifted) = \
                shifted_binary_cross_entropy_pt(Z, T,
                    torch.ones(T.size()), lc, eps)

        S = Z.detach().cpu().numpy()
        T = T.detach().cpu().numpy()
        mbs = T.shape[0]
        A = np.sum((np.round(S, 0) == T).astype(np.uint8))
        A_shifted = np.sum((np.round(S_shifted, 0) == T).astype(np.uint8))
        return A*1.0/mbs, A_shifted*1.0/mbs

    # pipeline training with pipeline
    def train_with_runtime(train_loader, r, optimizer, epoch):
        batch_time = AverageMeter()
        if is_last_stage():
            losses = AverageMeter()
            acc = AverageMeter()
            acc_shifted = AverageMeter()
            iter_acc = AverageMeter()
            iter_loss = AverageMeter()
        n = r.num_iterations(loader_size=len(train_loader))
        #print(n)
        # if nbatches is not None:
        #     n = min(n, nbatches)
        print("Total iterations: ", n)
        import sys; sys.stdout.flush()
        r.train(n)
        if not is_first_stage(): train_loader = None
        r.set_loader(train_loader)

        end = time.time()
        epoch_start_time = time.time()

        if args.no_input_pipelining:
            num_warmup_minibatches = 0
        else:
            num_warmup_minibatches = r.num_warmup_minibatches

        print("Let in %d warm-up minibatches" % num_warmup_minibatches)
        import sys; sys.stdout.flush()
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n-num_warmup_minibatches):
            r.run_forward()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss
                losses.update(loss.item(), output.size(0))
                iter_loss.update(loss.item(), output.size(0))
                # calculate the accuraccy
                tmpAcc, tmpAcc_shifted = cal_accuracy(output, target)
                acc.update(tmpAcc.item(), output.size(0))
                iter_acc.update(tmpAcc.item(), output.size(0))
                acc_shifted.update(tmpAcc_shifted.item(), output.size(0))

                batch_time.update(time.time()-end)
                end = time.time()
                epoch_time = (end - epoch_start_time) / 3600.0
                full_epoch_time = (epoch_time / float(i+1)) * float(n)
                if i % args.print_freq == 0:
                    print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                    'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                    'Accuracy: {acc.val:.4f} ({acc.avg:.4f}) (shifted: {acc_shifted.val:.4f})'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Period Acc: {iter_acc.val:.4f} ({iter_acc.avg:.4f})\t'
                    'Period Loss: {iter_loss.val:.4f} ({iter_loss.avg:.4f})'.format(
                    epoch, i, n, batch_time=batch_time,
                    epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                    loss=losses, acc=acc, acc_shifted=acc_shifted, iter_acc=iter_acc, iter_loss=iter_loss,
                    memory=(float(torch.cuda.memory_allocated()) / 10**9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    iter_acc.reset()
                    iter_loss.reset()
                    import sys; sys.stdout.flush()
            else:
                batch_time.update(time.time()-end)
                end = time.time()
                if i % args.print_freq == 0:
                    print('Train Epoch: [{0}][{1}/{2}]\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Memory: {memory:.3f} ({cached_memory:.3f})'.format(
                        epoch, i, n, batch_time=batch_time, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()
            # perform the backward pass
            if args.fp16:
                r.zero_grad()
            else:
                optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward()
            optimizer.load_new_params()
            optimizer.step()

        # finish the remaining backward passes
        for i in range(num_warmup_minibatches):
            optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward()
            optimizer.load_new_params()
            optimizer.step()
        # wait for all helper threads to complete

        r.wait()
        print("Train Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
        print("Train Epoch start time: %.3f, epoch end time: %.3f, elapsed time: %.3f" % (epoch_start_time, time.time(), time.time()-epoch_start_time))

    # pipeline validate with pipeline
    def test_with_runtime(test_loader, r, optimizer, epoch):
        batch_time = AverageMeter()
        if is_last_stage():
            losses = AverageMeter()
            acc = AverageMeter()
            acc_shifted = AverageMeter()
        n = r.num_iterations(loader_size=len(test_loader))
        # if nbatches_test is not None:
        #     n = min(n, nbatches_test)
        print("Total iterations: ", n)
        import sys; sys.stdout.flush()

        r.eval(n)
        if not is_first_stage(): test_loader = None
        r.set_loader(test_loader)

        end = time.time()
        epoch_start_time = time.time()

        if args.no_input_pipelining:
            num_warmup_minibatches = 0
        else:
            num_warmup_minibatches = r.num_warmup_minibatches

        print("Let in %d warm-up minibatches" % num_warmup_minibatches)
        with torch.no_grad():
            for i in range(num_warmup_minibatches):
                r.run_forward()

            for i in range(n-num_warmup_minibatches):
                r.run_forward()
                r.run_ack()

                if is_last_stage():
                    output, target, loss = r.output, r.target, r.loss
                    losses.update(loss.item(), output.size(0))
                    # calculate the accuraccy
                    tmpAcc, tmpAcc_shifted = cal_accuracy(output, target)
                    acc.update(tmpAcc.item(), output.size(0))
                    acc_shifted.update(tmpAcc_shifted.item(), output.size(0))

                    batch_time.update(time.time()-end)
                    end = time.time()
                    epoch_time = (end - epoch_start_time) / 3600.0
                    full_epoch_time = (epoch_time / float(i+1)) * float(n)
                    if i % args.print_freq == 0:
                        print('Test Epoch: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                        'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                        'Accuracy: {acc.val:.4f} (shifted: {acc_shifted.val:.4f})'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, i, n, batch_time=batch_time,
                        epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                        loss=losses, acc=acc, acc_shifted=acc_shifted,
                        memory=(float(torch.cuda.memory_allocated()) / 10**9),
                        cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                        import sys; sys.stdout.flush()

            if is_last_stage():
                print('Test Accuracy: {acc.avg:.4f} (shifted: {acc_shifted.avg:.4f})'
                      .format(acc=acc, acc_shifted=acc_shifted))
            # finish the remaining backward passes
            for i in range(num_warmup_minibatches):
                r.run_ack()

            # wait for all helper threads to complete
            r.wait()
            print("Test Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
            print("Test Epoch start time: %.3f, epoch end time: %.3f, elapsed time: %.3f" % (epoch_start_time, time.time(), time.time()-epoch_start_time))

    # training test
    print("Enter training: ", time.time(), "Iter Train: ", len(train_ld), ", Test: ", len(test_ld))
    for epoch in range(0, args.nepochs):
        if distributed_sampler:
            train_sampler.set_epoch(epoch)
        train_with_runtime(train_ld, r, optimizer, epoch)
        if len(test_ld) > 1:
            test_with_runtime(test_ld, r, optimizer, epoch)
    print("Leave training: ", time.time())
    exit()
