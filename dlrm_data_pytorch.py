# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i) Kaggle Display Advertising Challenge Dataset
#     https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/

from __future__ import absolute_import, division, print_function, unicode_literals

# others
from os import path
import bisect
import collections

from data_utils import *

# numpy
import numpy as np
from numpy import random as ra


# pytorch
import torch
from torch.utils.data import Dataset


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(Dataset):

    def __init__(self, dataset, randomize, split="train", raw_path="", pro_data=""):
        # dataset
        if dataset == "kaggle":
            days = 7
            df_exists = path.exists(str(pro_data))
            if df_exists:
                print("Reading from pre-processed data=%s" % (str(pro_data)))
                file = str(pro_data)
            else:
                print("Reading from raw data=%s" % (str(raw_path)))
                o_filename = "kaggleAdDisplayChallenge_processed"
                file = getKaggleCriteoAdData(raw_path, o_filename)
        elif dataset == "terabyte":
            raise(ValueError("Criteo Terabyte data-set is not supported"))

        # load and preprocess data
        with np.load(file) as data:
            X_int = data["X_int"] # continuous  feature
            X_cat = data["X_cat"] # categorical feature
            y = data["y"]         # target
            self.counts = data["counts"]
        self.m_den = X_int.shape[1]
        self.n_emb = len(self.counts)
        print("Sparse features = %d, Dense features = %d" % (self.n_emb, self.m_den))

        indices = np.arange(len(y))

        if split == "none":
            # randomize data
            if randomize == "total":
                indices = np.random.permutation(indices)
                print("Randomized indices...")

            self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in indices]

        else:
            indices = np.array_split(indices, days)

            # randomize each day's dataset
            if randomize == "day" or randomize == "total":
                for i in range(len(indices)):
                    indices[i] = np.random.permutation(indices[i])

            train_indices = np.concatenate(indices[:-1])
            test_indices = indices[-1]
            val_indices, test_indices = np.array_split(test_indices, 2)

            print("Defined %s indices..." % (split))

            # randomize all data in training set
            if randomize == "total":
                train_indices = np.random.permutation(train_indices)
                print("Randomized indices...")

            # create training, validation, and test sets
            if split == 'train':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in train_indices]
            elif split == 'val':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in test_indices]
            elif split == 'test':
                self.samples_list = [(X_int[i], X_cat[i], y[i]) for i in val_indices]

        print("Split data according to indices...")


    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        X_int, X_cat, y = self.samples_list[index]
        X_int, X_cat, y = self._default_preprocess(X_int, X_cat, y)

        return X_int, X_cat, y


    def _default_preprocess(self, X_int, X_cat, y):
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))
        # print("Converted to tensors...done!")

        return X_int, X_cat, y


    def __len__(self):
        return len(self.samples_list)


# uniform ditribution (input data)
class RandomDataset(Dataset):

    def __init__(self, m_den, ln_emb, data_size, num_batches, mini_batch_size, num_indices_per_lookup, num_indices_per_lookup_fixed, num_targets=1, round_targets=False, data_generation="random", trace_file="", enable_padding=False, reset_seed_on_access=False, rand_seed=0):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        if self.data_generation == "random":
            (X, lS_o, lS_i) = generate_uniform_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed
            )
        elif self.data_generation == "synthetic":
            (X, lS_o, lS_i) = generate_synthetic_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                self.trace_file,
                self.enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(n, self.num_targets, self.round_targets)

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def generate_random_data(
    m_den,
    ln_emb,
    data_size,
    num_batches,
    mini_batch_size,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    num_targets=1,
    round_targets=False,
    data_generation="random",
    trace_file="",
    enable_padding=False,
):
    nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
    if num_batches != 0:
        nbatches = num_batches
        data_size = nbatches * mini_batch_size
    # print("Total number of batches %d" % nbatches)

    # inputs
    lT = []
    lX = []
    lS_offsets = []
    lS_indices = []
    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        if data_generation == "random":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_uniform_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed
            )
        elif data_generation == "synthetic":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_synthetic_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                trace_file,
                enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + data_generation + " is not supported"
            )
        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_offsets.append(lS_emb_offsets)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_offsets, lS_indices, lT)

def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)

# uniform ditribution (input data)
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)

# synthetic distribution (input data)
def generate_synthetic_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    trace_file,
    enable_padding=False,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug prints
            # print("input")
            # print(line_accesses); print(list_sd); print(cumm_sd);
            # print(sparse_group_size)
            # approach 1: rand
            # r = trace_generate_rand(
            #     line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            # )
            # approach 2: lru
            r = trace_generate_lru(
                line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            )
            # WARNING: if the distribution in the file is not consistent
            # with embedding table dimensions, below mod guards against out
            # of range access
            sparse_group = np.unique(r).astype(np.int64)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int64)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    u = ra.rand(1)
    if i < max_i:
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        u = (1.0 - fi) * u + fi  # remap distribution support to exclude first value

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1


def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            line_accesses.pop(l - sd)
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace


def trace_generate_rand(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)  # !!!Unique,
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0
        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
        ztrace.append(mem_ref)

    return ztrace


def trace_profile(trace, enable_padding=False):
    # number of elements in the array (assuming 1D)
    # n = trace.size

    rstack = []  # S
    stack_distances = []  # SDS
    line_accesses = []  # L
    for x in trace:
        r = np.uint64(x / cache_line_size)
        l = len(rstack)
        try:  # found #
            i = rstack.index(r)
            # WARNING: I believe below is the correct depth in terms of meaning of the
            #          algorithm, but that is not what seems to be in the paper alg.
            #          -1 can be subtracted if we defined the distance between
            #          consecutive accesses (e.g. r, r) as 0 rather than 1.
            sd = l - i  # - 1
            # push r to the end of stack_distances
            stack_distances.insert(0, sd)
            # remove r from its position and insert to the top of stack
            rstack.pop(i)  # rstack.remove(r)
            rstack.insert(l - 1, r)
        except ValueError:  # not found #
            sd = 0  # -1
            # push r to the end of stack_distances/line_accesses
            stack_distances.insert(0, sd)
            line_accesses.insert(0, r)
            # push r to the top of stack
            rstack.insert(l, r)

    if enable_padding:
        # WARNING: notice that as the ratio between the number of samples (l)
        # and cardinality (c) of a sample increases the probability of
        # generating a sample gets smaller and smaller because there are
        # few new samples compared to repeated samples. This means that for a
        # long trace with relatively small cardinality it will take longer to
        # generate all new samples and therefore obtain full distribution support
        # and hence it takes longer for distribution to resemble the original.
        # Therefore, we may pad the number of new samples to be on par with
        # average number of samples l/c artificially.
        l = len(stack_distances)
        c = max(stack_distances)
        padding = int(np.ceil(l / c))
        stack_distances = stack_distances + [0] * padding

    return (rstack, stack_distances, line_accesses)


# auxiliary read/write routines
def read_trace_from_file(file_path):
    try:
        with open(file_path) as f:
            if args.trace_file_binary_type:
                array = np.fromfile(f, dtype=np.uint64)
                trace = array.astype(np.uint64).tolist()
            else:
                line = f.readline()
                trace = list(map(lambda x: np.uint64(x), line.split(", ")))
            return trace
    except Exception:
        print("ERROR: no input trace file has been provided")


def write_trace_to_file(file_path, trace):
    try:
        if args.trace_file_binary_type:
            with open(file_path, "wb+") as f:
                np.array(trace).astype(np.uint64).tofile(f)
        else:
            with open(file_path, "w+") as f:
                s = str(trace)
                f.write(s[1 : len(s) - 1])
    except Exception:
        print("ERROR: no output trace file has been provided")


def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
    except Exception:
        print("Wrong file or file path")
    # read unique accesses
    unique_accesses = [int(el) for el in lines[0].split(", ")]
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[1].split(", ")]
    cumm_sd = [float(el) for el in lines[2].split(", ")]

    return unique_accesses, list_sd, cumm_sd


def write_dist_to_file(file_path, unique_accesses, list_sd, cumm_sd):
    try:
        with open(file_path, "w") as f:
            # unique_acesses
            s = str(unique_accesses)
            f.write(s[1 : len(s) - 1] + "\n")
            # list_sd
            s = str(list_sd)
            f.write(s[1 : len(s) - 1] + "\n")
            # cumm_sd
            s = str(cumm_sd)
            f.write(s[1 : len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import sys
    import os
    import operator
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Generate Synthetic Distributions")
    parser.add_argument("--trace-file", type=str, default="./input/trace.log")
    parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    parser.add_argument("--trace-enable-padding", type=bool, default=False)
    parser.add_argument("--dist-file", type=str, default="./input/dist.log")
    parser.add_argument(
        "--synthetic-file", type=str, default="./input/trace_synthetic.log"
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--print-precision", type=int, default=5)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    ### read trace ###
    trace = read_trace_from_file(args.trace_file)
    # print(trace)

    ### profile trace ###
    (_, stack_distances, line_accesses) = trace_profile(
        trace, args.trace_enable_padding
    )
    stack_distances.reverse()
    line_accesses.reverse()
    # print(line_accesses)
    # print(stack_distances)

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(
        collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    )

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(
        map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    )  # k = tuple_x_k[1]
    cumm_sd = []  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    ### generate correspondinf synthetic ###
    # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    )
    # synthetic_trace = trace_generate_rand(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    write_trace_to_file(args.synthetic_file, synthetic_trace)
