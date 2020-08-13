# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Description: Modeling for different layers
# Author: Yanzhao Wu (yanzhaowu@fb.com)

def getLinearLayerFeatures(inSize, outSize, batchSize, linear_layer_flops,
                           linear_layer_flops_forward_factor=1.0, linear_layer_flops_backward_factor=1.0,
                           bias=True, dataSize=4.0):
    if bias:
        parameter_size = (inSize+1) * outSize * dataSize
        forward_computations = outSize
        # db = sum(dY^T, dim=1)
        backward_computations = batchSize * outSize
    else:
        parameter_size = inSize * outSize * dataSize
        forward_computations = 0
        backward_computations = 0

    # Y = xW + b
    forward_computations += batchSize * (inSize + inSize - 1) * outSize
    # dx = dY * W^T, dW = x^T * dY
    backward_computations += batchSize * (outSize + outSize - 1) * inSize + \
                             batchSize * inSize * outSize

    activation_size = batchSize * outSize * dataSize
    forward_compute_time = 1000.0 * forward_computations / (linear_layer_flops * linear_layer_flops_forward_factor)
    backward_compute_time = 1000.0 * backward_computations / (linear_layer_flops * linear_layer_flops_backward_factor)

    return forward_compute_time, backward_compute_time, activation_size, parameter_size


def getReLUFeatures(inSize, forward_time, backward_time=None, backward_time_factor=1.0):
    if not backward_time:
        backward_time = forward_time * backward_time_factor
    return forward_time, backward_time, inSize, 0.0


def getEmbeddingFeatures(inSize, outSize, batchSize, forward_time, backward_time=None, backward_time_factor=1.0, dataSize=4.0):
    if not backward_time:
        backward_time = forward_time * backward_time_factor
    activation_size = batchSize * outSize * dataSize
    parameter_size = inSize * outSize * dataSize

    return forward_time, backward_time, activation_size, parameter_size
