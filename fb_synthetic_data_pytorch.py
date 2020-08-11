#!/usr/bin/env python3

# Description: Generating synthetic data from the same distribution of the fb data.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import sys
import numpy as np
import os
import sys
import operator
from numpy import random as ra
from collections import deque
import collections
import bisect

# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1

from torch.utils.data import Dataset


def collate_wrapper_random(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


# auxiliary read routines
def read_trace_from_file(file_path, ignore_error=True, trace_file_binary_type=False):
    try:
        with open(file_path) as f:
            if trace_file_binary_type:
                array = np.fromfile(f, dtype=np.uint64)
                trace = array.astype(np.uint64).tolist()
            else:
                line = f.readline()
                if line == '':
                    trace = []
                else :
                    trace = list(map(lambda x: np.uint64(x), line.split(", ")))
            return trace
    except Exception as e:
        if ignore_error:
            print(f"Can not read '{file_path}' . {e}.")
        else:
            raise (e)


# auxiliary write routines
def write_trace_to_file(file_path, trace, trace_file_binary_type=False):
    try:
        if trace_file_binary_type:
            with open(file_path, "wb+") as f:
                np.array(trace).astype(np.uint64).tofile(f)
        else:
            with open(file_path, "w+") as f:
                s = str(trace)
                f.write(s[1 : len(s) - 1])
    except Exception as e:
        print(e, "\n Unable to write trace file ", file_path)


def trace_profile(trace, enable_padding=False):
    rstack = deque()  # S
    stack_distances = deque()  # SDS
    line_accesses = deque()  # L
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
            stack_distances.appendleft(sd)
            # remove r from its position and insert to the top of stack
            del rstack[i]  # rstack.remove(r)
            rstack.append(r)
        except ValueError:  # not found #
            sd = 0  # -1
            # push r to the end of stack_distances/line_accesses
            stack_distances.appendleft(sd)
            line_accesses.appendleft(r)
            # push r to the top of stack
            rstack.append(r)

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


def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False,
    max_value=-1
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = deque()
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses[0]
            del line_accesses[0]
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            del line_accesses[l - sd]
            line_accesses.append(line_ref)
        # save generated memory reference
        mem_ref = mem_ref % max_value if max_value > -1 else mem_ref
        ztrace.append(mem_ref)

    return ztrace


def syn_trace_from_trace(
        trace_file,
        syn_trace_len,
        prep_folder="stack_dists_line_aces/",
        trace_file_binary_type=False,
        trace_enable_padding=False,
        numpy_rand_seed=123,
        max_value=-1,
        print_precision=5
):

    ### some basic setup ###
    if numpy_rand_seed != -1:
        np.random.seed(numpy_rand_seed)
    np.set_printoptions(precision=print_precision)
    uni_name = '_'.join(trace_file[:-4].split('/'))

    ### profile trace ###
    dist_file = prep_folder + "dist_" + uni_name + ".npy"
    if not os.path.exists(dist_file):
        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)

        ### read trace ###
        trace = read_trace_from_file(trace_file)

        if trace == []:
            return []

        (_, stack_distances, line_accesses) = trace_profile(
            trace, trace_enable_padding
        )
        stack_distances.reverse()
        line_accesses.reverse()

        ### compute probability distribution ###
        # count items
        l = len(stack_distances)
        dc = sorted(
            collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
        )

        # create a distribution
        list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
        # dist_sd = list(
        #     map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
        # )  # k = tuple_x_k[1]
        cumm_sd = deque()  # np.cumsum(dc).tolist() #prefixsum
        for i, (_, k) in enumerate(dc):
            if i == 0:
                cumm_sd.append(k / float(l))
            else:
                # add the 2nd element of the i-th tuple in the dist_sd list
                cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

        ### write stack_distance and line_accesses to a file ###
        # dist_file_log = prep_folder + "dist_" + uni_name + '.log'
        # write_dist_to_file(dist_file_log, line_accesses, list_sd, cumm_sd)
        with open(dist_file, 'wb') as f:
            np.save(f, np.array(line_accesses))
            np.save(f, np.array(list_sd))
            np.save(f, np.array(cumm_sd))
        print('line_acs, list_sd, cumm_sd saved to ', dist_file)

    else:
        with open(dist_file, 'rb') as f:
            line_accesses = deque(np.load(f))
            list_sd = deque(np.load(f))
            cumm_sd = deque(np.load(f))
        print('line_acs, list_sd, cumm_sd loaded from ', dist_file)

    ### generate correspondinf synthetic ###
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, syn_trace_len, trace_enable_padding,
        max_value
    )
    # synthetic_trace = trace_generate_rand(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    # write synthetic trace to a file
    # synthetic_file = prep_folder + "syn_" + uni_name + '.log'
    # write_trace_to_file(synthetic_file, synthetic_trace)
    return synthetic_trace


def trace_generate_rand(
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
        ztrace.append(mem_ref)

    return ztrace


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


# fb synthetic distribution (input data)
def fb_generate_synthetic_input_batch(
    m_den,
    ln_emb,
    mini_batch_size,
    enable_padding=False,
    trace_folder="fb_traces/"
):
    # dense feature
    Xt = torch.tensor(ra.rand(mini_batch_size, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = deque()
    lS_emb_indices = deque()

    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices

    def is_available(trace_file):
        if not os.path.exists(trace_file):
            print("To generate trace_[i].log and size_trace_[i].log \
                files run emb_trace_writer.py ")
            sys.exit(f"{trace_file} is not available.")

    num_aval_traces = 0
    print_num_aval_traces = True
    for emb_tab_ind, size in enumerate(ln_emb):
        lS_batch_offsets = deque()
        lS_batch_indices = deque()

        trace_file = trace_folder + f'size_trace_{emb_tab_ind}.log'
        if not os.path.exists(trace_file):
            if emb_tab_ind == 0:
                print("To generate trace_[i].log and size_trace_[i].log \
                    files run emb_trace_writer.py ")
                sys.exit(f"{trace_file} is not available.")

            if print_num_aval_traces:
                print(f"Number of recognized trace files is {num_aval_traces}")
                print_num_aval_traces = False

            trace_file_id = emb_tab_ind % num_aval_traces
            trace_file_name = f'size_trace_{trace_file_id}.log'
            print(f"Trace file {trace_folder}{trace_file_name} "
                f"used instead of {trace_file}")
            trace_file = trace_folder + trace_file_name

        else:
            num_aval_traces += 1
            trace_file_id = emb_tab_ind

        lS_batch_sizes = np.array(syn_trace_from_trace(trace_file, mini_batch_size)).astype(np.int64)

        lS_batch_offsets = [0 for i in range(len(lS_batch_sizes))]
        for key, val in enumerate(lS_batch_sizes[:-1]):
            lS_batch_offsets[key + 1] = lS_batch_offsets[key] + val

        trace_file = trace_folder + f'trace_{trace_file_id}.log'
        is_available(trace_file)
        lS_batch_indices = np.array(
            syn_trace_from_trace(
                trace_file,
                lS_batch_offsets[-1] + lS_batch_sizes[-1],
                max_value=size
            )
        ).astype(np.int64)

        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, list(lS_emb_offsets), list(lS_emb_indices))


def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


class RandomDataset(Dataset):

    def __init__(
            self,
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
            reset_seed_on_access=False,
            rand_seed=0
    ):
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
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        if self.data_generation == "fb_synthetic":
            (X, lS_o, lS_i) = fb_generate_synthetic_input_batch(
                self.m_den,
                self.ln_emb,
                self.mini_batch_size,
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


def make_random_data_and_loader(args, ln_emb, m_den):

    train_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_seed=args.numpy_rand_seed
    )  # WARNING: generates a batch of lookups at once
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader

