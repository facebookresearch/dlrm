# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


class XlaEmbeddingBag(nn.Module):
    """
    nn.EmbeddingBag is not lowered just yet to xla.
    This performs the same functionality, in an xla compatible, sub-optimal way.

    Warning!: only works with constant offsets atm.
    """

    def __init__(self, n, m, mode, offset, *args, **kwargs):
        super(XlaEmbeddingBag, self).__init__()
        self.n = n
        self.m = m
        self.mode = mode
        self.offset = offset
        self.embtable = nn.Embedding(n, m, *args, **kwargs)

    def forward(self, sparse_index_group_batch, sparse_offset_group_batch):
        emb = self.embtable(sparse_index_group_batch)
        # XXX: only works w/ constant offset atm
        bsz = emb.size(0) // self.offset
        emb = emb.reshape(bsz, self.offset, *emb.size()[1:])
        reduce_fn = getattr(torch, self.mode)
        return reduce_fn(emb, axis=1)
        #return reduce_fn(self.embtable(_) for _ in inp_list)

    @property
    def weight(self):
        return self.embtable.weight
