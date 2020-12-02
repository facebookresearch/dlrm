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
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import collections

# others
# from os import path
import sys

import data_utils

# numpy
import numpy as np

# pytorch
import torch
from numpy import random as ra
from torch.utils.data import Dataset


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            'none': no randomization
#            'day': randomizes each day's data (only works if split = True)
#            'total': randomizes total dataset
# split (bool) : to split into train, test, validation data-sets


class CriteoDatasetWMemoryMap(Dataset):
    def __init__(
        self,
        dataset,
        max_ind_range,
        sub_sample_rate,
        randomize,
        split="train",
        raw_path="",
        pro_data="",
    ):
        # dataset
        # tar_fea = 1   # single target
        den_fea = 13  # 13 dense  features
        # spa_fea = 26  # 26 sparse features
        # tad_fea = tar_fea + den_fea
        # tot_fea = tad_fea + spa_fea
        if dataset == "kaggle":
            days = 7
        elif dataset == "terabyte":
            days = 24
        else:
            raise (ValueError("Data set option is not supported"))
        self.max_ind_range = max_ind_range

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
        self.npzfile = self.d_path + (
            (self.d_file + "_day") if dataset == "kaggle" else self.d_file
        )
        self.trafile = self.d_path + (
            (self.d_file + "_fea") if dataset == "kaggle" else "fea"
        )

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + list(total_per_file))
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)

        # setup data
        self.split = split
        if split == "none" or split == "train":
            self.day = 0
            self.max_day_range = days if split == "none" else days - 1
        elif split == "test" or split == "val":
            self.day = days - 1
            num_samples = self.offset_per_file[days] - self.offset_per_file[days - 1]
            self.test_size = int(np.ceil(num_samples / 2.0))
            self.val_size = num_samples - self.test_size
        else:
            sys.exit("ERROR: dataset split is neither none, nor train or test.")

        # load unique counts
        with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
            self.counts = data["counts"]
        self.m_den = den_fea  # X_int.shape[1]
        self.n_emb = len(self.counts)
        print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

        # Load the test data
        # Only a single day is used for testing
        if self.split == "test" or self.split == "val":
            # only a single day is used for testing
            fi = self.npzfile + "_{0}_reordered.npz".format(self.day)
            with np.load(fi) as data:
                self.X_int = data["X_int"]  # continuous  feature
                self.X_cat = data["X_cat"]  # categorical feature
                self.y = data["y"]  # target

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        if self.split == "none" or self.split == "train":
            # check if need to swicth to next day and load data
            if index == self.offset_per_file[self.day]:
                # print("day_boundary switch", index)
                self.day_boundary = self.offset_per_file[self.day]
                fi = self.npzfile + "_{0}_reordered.npz".format(self.day)
                # print('Loading file: ', fi)
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]  # target
                self.day = (self.day + 1) % self.max_day_range

            i = index - self.day_boundary
        elif self.split == "test" or self.split == "val":
            # only a single day is used for testing
            i = index + (0 if self.split == "test" else self.test_size)
        else:
            sys.exit("ERROR: dataset split is neither none, nor train or test.")

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def _default_preprocess(self, X_int, X_cat, y):
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        if self.max_ind_range > 0:
            X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
        else:
            X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))

        return X_int, X_cat, y

    def __len__(self):
        if self.split == "none":
            return self.offset_per_file[-1]
        elif self.split == "train":
            return self.offset_per_file[-2]
        elif self.split == "test":
            return self.test_size
        elif self.split == "val":
            return self.val_size
        else:
            sys.exit("ERROR: dataset split is neither none, nor train nor test.")


def collate_wrapper_criteo(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


# Conversion from offset to length
def offset_to_length_convertor(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def unpack_batch(b, data_gen, data_set):
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size())


def read_dataset(
    dataset,
    max_ind_range,
    sub_sample_rate,
    mini_batch_size,
    num_batches,
    randomize,
    split="train",
    raw_data="",
    processed_data="",
    memory_map=False,
    inference_only=False,
    test_mini_batch_size=1,
):
    # split the datafile into path and filename
    lstr = raw_data.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"
    d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
    # npzfile = d_path + ((d_file + "_day") if dataset == "kaggle" else d_file)
    # trafile = d_path + ((d_file + "_fea") if dataset == "kaggle" else "fea")

    # load
    print("Loading %s dataset..." % dataset)
    nbatches = 0
    file, days = data_utils.loadDataset(
        dataset,
        max_ind_range,
        sub_sample_rate,
        randomize,
        split,
        raw_data,
        processed_data,
        memory_map,
    )

    if memory_map:
        # WARNING: at this point the data has been reordered and shuffled across files
        # e.g. day_<number>_reordered.npz, what remains is simply to read and feed
        # the data from each file, going in the order of days file-by-file, to the
        # model during training.
        train_data = CriteoDatasetWMemoryMap(
            dataset,
            max_ind_range,
            sub_sample_rate,
            randomize,
            "train",
            raw_data,
            processed_data,
        )

        test_data = CriteoDatasetWMemoryMap(
            dataset,
            max_ind_range,
            sub_sample_rate,
            randomize,
            "test",
            raw_data,
            processed_data,
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=test_mini_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

        return train_data, train_loader, test_data, test_loader

    else:
        # load and preprocess data
        with np.load(file) as data:
            X_int = data["X_int"]
            X_cat = data["X_cat"]
            y = data["y"]
            counts = data["counts"]

        # get a number of samples per day
        total_file = d_path + d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]

        # transform
        (
            X_cat_train,
            X_int_train,
            y_train,
            X_cat_val,
            X_int_val,
            y_val,
            X_cat_test,
            X_int_test,
            y_test,
        ) = data_utils.transformCriteoAdData(
            X_cat, X_int, y, days, split, randomize, total_per_file
        )
        ln_emb = counts
        m_den = X_int_train.shape[1]
        n_emb = len(counts)
        print("Sparse features = %d, Dense features = %d" % (n_emb, m_den))

        # adjust parameters
        def assemble_samples(X_cat, X_int, y, max_ind_range, print_message):
            if max_ind_range > 0:
                X_cat = X_cat % max_ind_range

            nsamples = len(y)
            data_size = nsamples
            # using floor is equivalent to dropping last mini-batch (drop_last = True)
            nbatches = int(np.floor((data_size * 1.0) / mini_batch_size))
            print(print_message)
            if num_batches != 0 and num_batches < nbatches:
                print(
                    "Limiting to %d batches of the total % d batches"
                    % (num_batches, nbatches)
                )
                nbatches = num_batches
            else:
                print("Total number of batches %d" % nbatches)

            # data main loop
            lX = []
            lS_lengths = []
            lS_indices = []
            lT = []
            for j in range(0, nbatches):
                # number of data points in a batch
                print("Reading in batch: %d / %d" % (j + 1, nbatches), end="\r")
                n = min(mini_batch_size, data_size - (j * mini_batch_size))
                # dense feature
                idx_start = j * mini_batch_size
                lX.append((X_int[idx_start : (idx_start + n)]).astype(np.float32))
                # Targets - outputs
                lT.append(
                    (y[idx_start : idx_start + n]).reshape(-1, 1).astype(np.int32)
                )
                # sparse feature (sparse indices)
                lS_emb_indices = []
                # for each embedding generate a list of n lookups,
                # where each lookup is composed of multiple sparse indices
                for size in range(n_emb):
                    lS_batch_indices = []
                    for _b in range(n):
                        # num of sparse indices to be used per embedding, e.g. for
                        # store lengths and indices
                        lS_batch_indices += (
                            (X_cat[idx_start + _b][size].reshape(-1)).astype(np.int32)
                        ).tolist()
                    lS_emb_indices.append(lS_batch_indices)
                lS_indices.append(lS_emb_indices)
                # Criteo Kaggle data it is 1 because data is categorical
                lS_lengths.append(
                    [(list(np.ones(n).astype(np.int32))) for _ in range(n_emb)]
                )
            print("\n")

            return nbatches, lX, lS_lengths, lS_indices, lT

        # adjust training data
        (nbatches, lX, lS_lengths, lS_indices, lT) = assemble_samples(
            X_cat_train, X_int_train, y_train, max_ind_range, "Training data"
        )

        # adjust testing data
        (nbatches_t, lX_t, lS_lengths_t, lS_indices_t, lT_t) = assemble_samples(
            X_cat_test, X_int_test, y_test, max_ind_range, "Testing data"
        )
    # end if memory_map

    return (
        nbatches,
        lX,
        lS_lengths,
        lS_indices,
        lT,
        nbatches_t,
        lX_t,
        lS_lengths_t,
        lS_indices_t,
        lT_t,
        ln_emb,
        m_den,
    )


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

    # inputs and targets
    lT = []
    lX = []
    lS_lengths = []
    lS_indices = []
    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        if data_generation == "random":
            (Xt, lS_emb_lengths, lS_emb_indices) = generate_uniform_input_batch(
                m_den, ln_emb, n, num_indices_per_lookup, num_indices_per_lookup_fixed
            )
        elif data_generation == "synthetic":
            (Xt, lS_emb_lengths, lS_emb_indices) = generate_synthetic_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                trace_file,
                enable_padding,
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + data_generation + " is not supported"
            )
        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_lengths.append(lS_emb_lengths)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_lengths, lS_indices, lT)


def generate_random_output_batch(n, num_targets=1, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.int32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return P


# uniform ditribution (input data)
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
):
    # dense feature
    Xt = ra.rand(n, m_den).astype(np.float32)

    # sparse feature (sparse indices)
    lS_emb_lengths = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_lengths = []
        lS_batch_indices = []
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int32(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int32(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int32))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_lengths += [sparse_group_size]
            lS_batch_indices += sparse_group.tolist()
        lS_emb_lengths.append(lS_batch_lengths)
        lS_emb_indices.append(lS_batch_indices)

    return (Xt, lS_emb_lengths, lS_emb_indices)


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
    Xt = ra.rand(n, m_den).astype(np.float32)

    # sparse feature (sparse indices)
    lS_emb_lengths = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_lengths = []
        lS_batch_indices = []
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int32(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int32(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug print
            # print('input')
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
            # WARNING: if the distribution in the file is not consistent with
            # embedding table dimensions, below mod guards against out of
            # range access
            sparse_group = np.unique(r).astype(np.int32)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int32)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int32))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_lengths += [sparse_group_size]
            lS_batch_indices += sparse_group.tolist()
        lS_emb_lengths.append(lS_batch_lengths)
        lS_emb_indices.append(lS_batch_indices)

    return (Xt, lS_emb_lengths, lS_emb_indices)


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
