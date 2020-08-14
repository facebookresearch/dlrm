# Profiling Deep Learning Recommendation Models
# Author: Yanzhao Wu (yanzhaowu@fb.com)

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json

# data generation
import dlrm_data_pytorch as dp

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
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
# mixed-dimension trick
from tricks.md_embedding_bag import md_solver
# DLRM_Net model
from models.DLRM_Net import DLRM_Net

import sklearn.metrics

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

# pipeline profiling (profiling libraries)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
#import sys; sys.path = [".."] + sys.path
from profiler.profile_utils import AverageMeter
import profiler.torchmodules.torchgraph as torchgraph
import profiler.torchmodules.torchlogger as torchlogger
import profiler.torchmodules.torchprofiler as torchprofiler
import profiler.torchmodules.torchsummary as torchsummary

from collections import OrderedDict
import os

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
        description="Train & Profile Deep Learning Recommendation Model (DLRM)"
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

    # pipeline profiling arguments
    parser.add_argument("--enable-pipeline-profiling", action="store_true", default=False)
    parser.add_argument('--pipeline-profile-directory', default="myprofiles/",
                        help="Pipeline Profile directory")
    parser.add_argument('--model-name', default="dlrm",
                        help="The model name")
    parser.add_argument('-v', '--verbose', action='store_true',
                    help="Controls verbosity while pipeline profiling")

    args = parser.parse_args()

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
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
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
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

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

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
    )

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        """
        if ngpus > 1:
            # approach 1: DataParallel
            # approach 1.1: entire model
            dlrm = torch.nn.DataParallel(dlrm, device_ids=list(range(ngpus)))
            # approach 1.2: only mlps
            dlrm.emb_l = nn.ModuleList(map(lambda e: e.to(device), dlrm.emb_l))
            dlrm.bot_l = torch.nn.DataParallel(dlrm.bot_l,device_ids=list(range(ngpus)))
            dlrm.top_l = torch.nn.DataParallel(dlrm.top_l,device_ids=list(range(ngpus)))
        """
        """
        if ngpus > 1:
            # approach 2: DistributedDataParallel
            # init processes
            # WARNING: set environment variables or use init_method argument below
            # approach 2.1: environment variables
            os.environ['MASTER_ADDR'] = '127.0.0.1' # IP address of current node
            os.environ['MASTER_PORT'] = '29500'     # FREEPORT
            torch.distributed.init_process_group(
                backend="nccl", # nccl or gloo
                rank=0,
                world_size=1)
            # approach 2.2: init_method
            torch.distributed.init_process_group(
                backend="nccl", # nccl or gloo
                init_method='tcp://127.0.0.1:29500',
                rank=0,
                world_size=1)
            # call DistributedDataParallel
            dlrm = torch.nn.parallel.DistributedDataParallel(
                dlrm, device_ids=list(range(ngpus)))
        """
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
                m_spa, ln_emb, args.weighted_pooling
            )
        else:
            if dlrm.weighted_pooling == "fixed":
                for k, w in enumerate(dlrm.v_W_l):
                    dlrm.v_W_l[k] = w.cuda()

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

    if not args.inference_only:
        # specify the optimizer algorithm
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)

    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(X, lS_o, lS_i, use_gpu, device):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return dlrm(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return dlrm(X, lS_o, lS_i)

    def loss_fn_wrap(Z, T, use_gpu, device):
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0
    total_accu_shifted = 0
    lifetime_loss_shifted = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0
        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for post-training quantization during the inference.
        # By default we don't do the quantization: quantize_with_bit == 32 (FP32)
        assert(
            args.quantize_with_bit == 8
            or args.quantize_with_bit == 16
            or args.quantize_with_bit == 32
        )
        if args.quantize_with_bit != 32:
            if args.quantize_with_bit == 8:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
            # print(dlrm)

    print("time/loss/accuracy (if enabled):")
    #tb_file = './' + args.tensor_board_filename
    #writer = SummaryWriter(tb_file)

    # pipeline profiling
    if args.enable_pipeline_profiling and use_gpu:
        # get the first batch for summary
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W = unpack_batch(inputBatch)
            if j >= 0:
                break
        Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device) # pipeline: first iteration for initialization.
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
            else lS_i.to(device)
        lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
            else lS_o.to(device)
        pipeline_p_summary = torchsummary.summary(
            model=dlrm, module_whitelist=[],
            model_input=(X.to(device),lS_o, lS_i),
            verbose=args.verbose, device="cuda" # please set the CUDA_VISIBLE_DEVICES=0
        )
        def profile_train(train_loader, model):
            batch_time_meter = AverageMeter()
            data_time_meter = AverageMeter()
            NUM_STEPS_TO_PROFILE = 100  # profile 100 steps or minibatches

            layer_timestamps = []
            data_times = []

            iteration_timestamps = []
            opt_step_timestamps = []
            data_timestamps = []

            start_time = time.time()
            for i, inputBatch in enumerate(train_loader):
                data_pid = os.getpid()
                data_time = time.time() - start_time
                data_time_meter.update(data_time)
                X, lS_o, lS_i, T, W = unpack_batch(inputBatch)

                with torchprofiler.Profiling(model, module_whitelist=[]) as p:
                    # forward pass
                    Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device)
                    # loss
                    E = loss_fn_wrap(Z, T, use_gpu, device)
                    optimizer.zero_grad()
                    # backward pass
                    E.backward()
                    optimizer_step_start = time.time()
                    # optimizer
                    optimizer.step()
                    end_time = time.time()
                    iteration_time = end_time - start_time
                    batch_time_meter.update(iteration_time)

                    if i >= NUM_STEPS_TO_PROFILE:
                        break
                p_str = str(p)
                layer_timestamps.append(p.processed_times())
                data_times.append(data_time)

                if args.verbose:
                    print('End-to-end time: {batch_time.val:.3f} s ({batch_time.avg:.3f} s)'.format(
                        batch_time=batch_time_meter))

                iteration_timestamps.append({"start": start_time * 1000 * 1000,
                                            "duration": iteration_time * 1000 * 1000})
                opt_step_timestamps.append({"start": optimizer_step_start * 1000 * 1000,
                                            "duration": (end_time - optimizer_step_start) * 1000 * 1000, "pid": os.getpid()})
                data_timestamps.append({"start":  start_time * 1000 * 1000,
                                        "duration": data_time * 1000 * 1000, "pid": data_pid})

                start_time = time.time()

            layer_times = []
            tot_accounted_time = 0.0
            if args.verbose:
                print("\n==========================================================")
                print("Layer Type    Forward Time (ms)    Backward Time (ms)")
                print("==========================================================")

            for i in range(len(layer_timestamps[0])):
                layer_type = str(layer_timestamps[0][i][0])
                layer_forward_time_sum = 0.0
                layer_backward_time_sum = 0.0
                for j in range(len(layer_timestamps)):
                    layer_forward_time_sum += (layer_timestamps[j][i][2] / 1000)
                    layer_backward_time_sum += (layer_timestamps[j][i][5] / 1000)
                layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                            layer_backward_time_sum / len(layer_timestamps)))
                if args.verbose:
                    print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
                tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

            print()
            print("Total accounted time: %.3f ms" % tot_accounted_time)
            return layer_times, (sum(data_times) * 1000.0) / len(data_times)
        # end of profile_train
        per_layer_times, data_time = profile_train(train_ld, dlrm)

        pipeline_p_summary_i = 0
        per_layer_times_i = 0
        while pipeline_p_summary_i < len(pipeline_p_summary) and per_layer_times_i < len(per_layer_times):
            pipeline_p_summary_elem = pipeline_p_summary[pipeline_p_summary_i]
            per_layer_time = per_layer_times[per_layer_times_i]
            if str(pipeline_p_summary_elem['layer_name']) != str(per_layer_time[0]):
                pipeline_p_summary_elem['forward_time'] = 0.0
                pipeline_p_summary_elem['backward_time'] = 0.0
                pipeline_p_summary_i += 1
                continue
            pipeline_p_summary_elem['forward_time'] = per_layer_time[1]
            pipeline_p_summary_elem['backward_time'] = per_layer_time[2]
            pipeline_p_summary_i += 1
            per_layer_times_i += 1
        # pipeline_p_summary.append(OrderedDict())
        # pipeline_p_summary[-1]['layer_name'] = 'Input'
        # pipeline_p_summary[-1]['forward_time'] = data_time
        # pipeline_p_summary[-1]['backward_time'] = 0.0
        # pipeline_p_summary[-1]['nb_params'] = 0.0
        # pipeline_p_summary[-1]['output_shape'] = [args.mini_batch_size] #TODO: correct output_shape

        def create_graph(model, train_loader, summary, directory):
            """Given a model, creates and visualizes the computation DAG
            of the model in the passed-in directory."""
            graph_creator = torchgraph.GraphCreator(model, summary, module_whitelist=[])
            graph_creator.hook_modules(model)
            for i, inputBatch in enumerate(train_loader):
                X, lS_o, lS_i, T, W = unpack_batch(inputBatch)
                # forward pass
                Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device)
                if i >= 0:
                    break
            graph_creator.unhook_modules()
            graph_creator.persist_graph(directory)

        create_graph(dlrm, train_ld, pipeline_p_summary,
                     os.path.join(args.pipeline_profile_directory, args.model_name))
        print("...done!")
        quit()
