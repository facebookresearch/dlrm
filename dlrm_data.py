# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: delivering inputs and targets for the dlrm benchmark
# The inpts and outputs are used according to the following two option(s)
# 1) random distribution, generated and loaded based on uniform distribution
# 2) synthetic data, the synthetic pre-generated data would be loaded.

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
from numpy import random as ra
import torch
from torch.utils.data import Dataset  # , RandomSampler


class RandomDataset(Dataset):
    """ Uniform distribution """
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
        if self.data_generation == "random":
            (X, lS_o, lS_i) = generate_uniform_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed
            )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(n, self.num_targets, self.round_targets)

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def collate_wrapper_random(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


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
    #Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))
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


class SyntheticDataset(Dataset):

    def __init__(
        self,
        mini_batch_size,
        ln_emb,
        nbatches=1,
        synthetic_data_folder="./synthetic_data/syn_data_bs65536/",
    ):
        self.synthetic_data_folder = synthetic_data_folder
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.ln_emb = ln_emb

        self.X = torch.load(f"{self.synthetic_data_folder}/X_0.pt")
        self.lS_o = torch.load(f"{self.synthetic_data_folder}/lS_o_0.pt")
        self.lS_i = torch.load(f"{self.synthetic_data_folder}/lS_i_0.pt")
        self.T = torch.load(f"{self.synthetic_data_folder}/T_0.pt")
        # print('data loader initiated ...')

    def __getitem__(self, index):
        sInd = index * self.mini_batch_size
        eInd = sInd + self.mini_batch_size
        if sInd >= len(self.X):
            sys.exit(f' mini_batch_size({self.mini_batch_size}) * '
                f'num_batches({self.num_batches}) has to be less'
                f' than size of data({len(self.X)})'
            )
        X = self.X[sInd:eInd]
        lS_o = [i[:][sInd:eInd] - i[:][sInd] for i in self.lS_o]

        if eInd < len(self.lS_o[0]):
            lS_i = [val[self.lS_o[ind][sInd]:self.lS_o[ind][eInd]] for ind, val in enumerate(self.lS_i)]
        elif sInd < len(self.lS_o[0]):
            lS_i = [val[self.lS_o[ind][sInd]:] for ind, val in enumerate(self.lS_i)]
        for i in range(len(lS_i)):
            bound = self.ln_emb[i]
            if not bound == 26000000:
                lS_i[i] %= bound
            
        T = self.T[sInd:eInd]
        return (X, lS_o, lS_i, T)

    def __len__(self):
        return self.num_batches


def synthetic_data_loader(args, ln_emb, m_den):

    train_data = SyntheticDataset(
        args.mini_batch_size,
        ln_emb,
        nbatches=args.num_batches,
        synthetic_data_folder=args.synthetic_data_folder,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,
    )
    return train_data, train_loader


def data_loader(args, ln_emb, m_den):
    data_gens = {"random": make_random_data_and_loader,
                 "synthetic": synthetic_data_loader,
    }
    train_data, train_ld = data_gens[args.data_generation](args, ln_emb, m_den)

    return train_data, train_ld
