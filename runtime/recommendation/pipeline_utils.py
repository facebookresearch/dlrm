# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Description: This file contains a set of utility functions for pipeline runtime.
# Author: Yanzhao Wu (yanzhaowu@fb.com)

import torch
import torch.nn as nn


# The loss function wrapper class for DLRM
class LossFnWrapper(nn.Module):
    def __init__(self, _loss_fn, _use_gpu, _device, _args, _loss_ws=None):
        super(LossFnWrapper, self).__init__()
        self.loss_fn = _loss_fn
        self.use_gpu = _use_gpu
        self.device = _device
        self.args = _args
        self.loss_ws = _loss_ws

    def forward(self, input, target):
        #clamp output
        Z = torch.clamp(input, min=0.0, max=1.0)
        T = target.detach()
        #print("Z, T dtype:", Z.dtype, T.dtype)
        #print("In loss fn: ", Z, T)
        # modified from the loss_fn_wrap function in DLRM
        if self.args.loss_function == "mse" or self.args.loss_function == "bce":
            if self.use_gpu:
                loss_value = self.loss_fn(Z, T.to(self.device))
            else:
                loss_value = self.loss_fn(Z, T)
        elif self.args.loss_function == "wbce":
            if self.use_gpu:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T).to(self.device)
                loss_fn_ = self.loss_fn(Z, T.to(self.device))
            else:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = self.loss_fn(Z, T.to(self.device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            loss_value = loss_sc_.mean()
        return loss_value
