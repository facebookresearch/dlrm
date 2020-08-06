# Description: using torch internal koski reader to write two traces
# for each embedding table. The first one is trace_{i}.log which represents
# indices of each sample for embedding table i. The second one is size_trace_{i}.log
# which stores the number of indices for each sample for embedding table i.
# These traces would be used in fb synthetic data generation which imitate fb data
# for benchmarking DLRM in external infrastructures.


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import builtins
import argparse
import os

# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# pytorch
import torch

# The following import is needed only for runs that use production data
# For those runs, the proper dependencies should be set up before the run.
try:
    import fb_data_pytorch as fbdata
except ImportError:
    print("Production libs are not set up.")
    print("Note: Not all runs need production libs.")
    pass

exc = getattr(builtins, "IOError", "FileNotFoundError")


if __name__ == "__main__":
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
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    parser.add_argument("--hype-flag", action="store_true", default=False)
    parser.add_argument("--cluster-fb-preproc", action="store_true", default=False)
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
    parser.add_argument("--fb-test-sample-limit", type=int, default=5000)
    parser.add_argument("--fb-train-sample-limit", type=int, default=500000)
    parser.add_argument("--fb-test-ds", type=str, default="")
    parser.add_argument("--fb-train-ds", type=str, default="")
    parser.add_argument("--fb-stream-train-samples", action="store_true", default=False)
    parser.add_argument("--fb-stream-test-samples", action="store_true", default=False)
    parser.add_argument("--fb-stream-prefetch-size", type=int, default=128)
    parser.add_argument("--fb-preprocess", action="store_true", default=False)
    parser.add_argument("--fb-report-ne", action="store_true", default=False)
    parser.add_argument("--fb-window-size", type=int, default=-1)
    parser.add_argument("--fb-loss-bias", type=float, default=1.0)  # shifted BCE loss c
    parser.add_argument("--fb-run-id", type=int, default=-1)
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
    parser.add_argument("--trace-folder", type=str, default="fb_traces/")
    args = parser.parse_args()
    print(args)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if args.data_generation == "dataset":

        if (args.data_set == "fb"):
            train_data, train_ld, test_data, test_ld, table_feature_map = \
                fbdata.make_fb_data_and_loaders(args)
            nbatches = args.num_batches if args.num_batches > 0 \
                else train_data.num_batches
            # TODO: Handle the cases where fb_train_sample_limit is set to more
            # than the number of samples in the table.
            nbatches_test = test_data.num_batches
        else:  # criteo kaggle or terabyte
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
        else:
            ln_emb = np.array(ln_emb)

        m_den = train_data.m_den
        ln_bot[0] = m_den

    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    # For FB data, each batch has a label tensor packed, while the others don't.
    # The following function is a wrapper to avoid checking this multiple times in th
    # loop below.
    def unpack_batch(b):
        if args.data_generation == "dataset" and args.data_set == "fb":
            # Experiment with weighted samples
            return b[0], b[1], b[2], b[3], b[4]
        else:
            # Experiment with unweighted samples
            return b[0], b[1], b[2], b[3], torch.ones(b[3].size())

    ### parse command line arguments ###
    num_fea = ln_emb.size + 1  # num sparse + num dense features

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
    print(
        "# of embeddings (= # of sparse features) "
        + str(ln_emb.size)
        + ", with dimensions "
        + "x:"
    )
    print(ln_emb)

    f_traces = [-1 for i in range(ln_emb.size)]
    f_size_trace = [-1 for i in range(ln_emb.size)]
    if not os.path.exists(args.trace_folder):
        os.mkdir(args.trace_folder)

    for i in range(ln_emb.size):
        f_traces[i] = open(f'{args.trace_folder}/trace_{i}.log', 'w')
        f_size_trace[i] = open(f'{args.trace_folder}/size_trace_{i}.log', 'w')

    def f(x):
        return x.item()

    for j, inputBatch in enumerate(train_ld):
        X, lS_o, lS_i, T, W = unpack_batch(inputBatch)

        # sparse index traces
        for table_ind, S_i in enumerate(lS_i):

            for item_ind, item in enumerate(S_i):
                elm = f"{item}, "
                if j == nbatches - 1 and item_ind == len(S_i) - 1:  # the final last item
                    elm = f"{item}"

                f_traces[table_ind].write(elm)

        # size traces
        for table_ind, S_o in enumerate(lS_o):

            s_i = lS_i[table_ind]
            for ind, _item in enumerate(S_o):
                if ind == len(S_o) - 1:  # the last item in this batch
                    size_trace_elm = f"{len(s_i) - f(S_o[-1])}, "
                    if j == nbatches - 1 :  # the final last item
                        size_trace_elm = f"{len(s_i) - f(S_o[-1])}"
                else:
                    itm = f(S_o[ind + 1]) - f(S_o[ind])
                    size_trace_elm = str(itm) + ', '

                f_size_trace[table_ind].write(size_trace_elm)

        if (j + 1) % args.print_freq == 0:
            print(f"mini-batch: {j + 1}")

    # closing trace files
    for i in range(ln_emb.size):
        f_traces[i].close()
        f_size_trace[i].close()

    print(f"Trace and size_trace files are in the folder {args.trace_folder}")
