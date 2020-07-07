# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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

import functools

# others
import operator
import time
import copy

# data generation
import dlrm_data_caffe2 as dc

# numpy
import numpy as np
import sklearn.metrics

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx
import caffe2.python.onnx.frontend

# caffe2
from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, dyndep, model_helper, net_drawer, workspace
# from caffe2.python.predictor import mobile_exporter

"""
# auxiliary routine used to split input on the mini-bacth dimension
def where_to_split(mini_batch_size, ndevices, _add_leftover=False):
    n = (mini_batch_size + ndevices - 1) // ndevices  # ceiling
    l = mini_batch_size - n * (ndevices - 1)  # leftover
    s = [n] * (ndevices - 1)
    if _add_leftover:
        ls += [l if l > 0 else n]
    return ls
"""


### define dlrm in Caffe2 ###
class DLRM_Net(object):
    def FeedBlobWrapper(self, tag, val, add_prefix=True, split=False, device_id=-1):
        if self.ndevices > 1 and add_prefix:
            if split:
                # split across devices
                mini_batch_size = val.shape[0]
                # approach 1: np and caffe2 operators assume the mini-batch size is
                # divisible exactly by the number of available devices
                if mini_batch_size % self.ndevices != 0:
                    sys.exit("ERROR: caffe2 net assumes that the mini_batch_size "
                             + str(mini_batch_size)
                             + " is evenly divisible by the number of available devices"
                             + str(self.ndevices))
                vals = np.split(val, self.ndevices, axis=0)
                """
                # approach 2: np and caffe2 operators do not assume exact divisibility
                if args.mini_batch_size != mini_batch_size:
                    sys.exit("ERROR: caffe2 net was prepared for mini-batch size "
                             + str(args.mini_batch_size)
                             + " which is different from current mini-batch size "
                             + str(mini_batch_size) + " being passed to it. "
                             + "This is common for the last mini-batch, when "
                             + "mini-batch size does not evenly divided the number of "
                             + "elements in the data set.")
                ls = where_to_split(mini_batch_size, self.ndevices)
                vals = np.split(val, ls, axis=0)
                """
                # feed to multiple devices
                for d in range(self.ndevices):
                    tag_on_device = "gpu_" + str(d) + "/" + tag
                    _d = core.DeviceOption(workspace.GpuDeviceType, d)
                    workspace.FeedBlob(tag_on_device, vals[d], device_option=_d)
            else:
                # feed to multiple devices
                for d in range(self.ndevices):
                    tag_on_device = "gpu_" + str(d) + "/" + tag
                    _d = core.DeviceOption(workspace.GpuDeviceType, d)
                    workspace.FeedBlob(tag_on_device, val, device_option=_d)
        else:
            # feed to a single device (named or not)
            if device_id >= 0:
                _d = core.DeviceOption(workspace.GpuDeviceType, device_id)
                workspace.FeedBlob(tag, val, device_option=_d)
            else:
                workspace.FeedBlob(tag, val)

    def FetchBlobWrapper(self, tag, add_prefix=True, reduce_across=None, device_id=-1):
        if self.ndevices > 1 and add_prefix:
            # fetch from multiple devices
            vals = []
            for d in range(self.ndevices):
                if tag.__class__ == list:
                    tag_on_device = tag[d]
                else:
                    tag_on_device = "gpu_" + str(0) + "/" + tag
                val = workspace.FetchBlob(tag_on_device)
                vals.append(val)
            # reduce across devices
            if reduce_across == "add":
                return functools.reduce(operator.add, vals)
            elif reduce_across == "concat":
                return np.concatenate(vals)
            else:
                return vals
        else:
            # fetch from a single device (named or not)
            if device_id >= 0:
                tag_on_device = "gpu_" + str(device_id) + "/" + tag
                return workspace.FetchBlob(tag_on_device)
            else:
                return workspace.FetchBlob(tag)

    def AddLayerWrapper(self, layer, inp_blobs, out_blobs,
                        add_prefix=True, reset_grad=False, **kwargs):
        # auxiliary routine to adjust tags
        def adjust_tag(blobs, on_device):
            if blobs.__class__ == str:
                _blobs = on_device + blobs
            elif blobs.__class__ == list:
                _blobs = list(map(lambda tag: on_device + tag, blobs))
            else:  # blobs.__class__ == model_helper.ModelHelper or something else
                _blobs = blobs
            return _blobs

        if self.ndevices > 1 and add_prefix:
            # add layer on multiple devices
            ll = []
            for d in range(self.ndevices):
                # add prefix on_device
                on_device = "gpu_" + str(d) + "/"
                _inp_blobs = adjust_tag(inp_blobs, on_device)
                _out_blobs = adjust_tag(out_blobs, on_device)
                # WARNING: reset_grad option was exlusively designed for WeightedSum
                #         with inp_blobs=[w, tag_one, "", lr], where "" will be replaced
                if reset_grad:
                    w_grad = self.gradientMap[_inp_blobs[0]]
                    _inp_blobs[2] = w_grad
                # add layer to the model
                with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, d)):
                    if kwargs:
                        new_layer = layer(_inp_blobs, _out_blobs, **kwargs)
                    else:
                        new_layer = layer(_inp_blobs, _out_blobs)
                ll.append(new_layer)
            return ll
        else:
            # add layer on a single device
            # WARNING: reset_grad option was exlusively designed for WeightedSum
            #          with inp_blobs=[w, tag_one, "", lr], where "" will be replaced
            if reset_grad:
                w_grad = self.gradientMap[inp_blobs[0]]
                inp_blobs[2] = w_grad
            # add layer to the model
            if kwargs:
                new_layer = layer(inp_blobs, out_blobs, **kwargs)
            else:
                new_layer = layer(inp_blobs, out_blobs)
            return new_layer

    def create_mlp(self, ln, sigmoid_layer, model, tag):
        (tag_layer, tag_in, tag_out) = tag

        # build MLP layer by layer
        layers = []
        weights = []
        for i in range(1, ln.size):
            n = ln[i - 1]
            m = ln[i]

            # create tags
            tag_fc_w = tag_layer + ":::" + "fc" + str(i) + "_w"
            tag_fc_b = tag_layer + ":::" + "fc" + str(i) + "_b"
            tag_fc_y = tag_layer + ":::" + "fc" + str(i) + "_y"
            tag_fc_z = tag_layer + ":::" + "fc" + str(i) + "_z"
            if i == ln.size - 1:
                tag_fc_z = tag_out
            weights.append(tag_fc_w)
            weights.append(tag_fc_b)

            # initialize the weights
            # approach 1: custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            b = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            self.FeedBlobWrapper(tag_fc_w, W)
            self.FeedBlobWrapper(tag_fc_b, b)
            # approach 2: caffe2 xavier
            # W = self.AddLayerWrapper(
            #     model.param_init_net.XavierFill,
            #     [],
            #     tag_fc_w,
            #     shape=[m, n]
            # )
            # b = self.AddLayerWrapper(
            #     model.param_init_net.ConstantFill,
            #     [],
            #     tag_fc_b,
            #     shape=[m]
            # )
            # save the blob shapes for latter (only needed if onnx is requested)
            if self.save_onnx:
                self.onnx_tsd[tag_fc_w] = (onnx.TensorProto.FLOAT, W.shape)
                self.onnx_tsd[tag_fc_b] = (onnx.TensorProto.FLOAT, b.shape)

            # approach 1: construct fully connected operator using model.net
            fc = self.AddLayerWrapper(
                model.net.FC, [tag_in, tag_fc_w, tag_fc_b], tag_fc_y
            )
            # approach 2: construct fully connected operator using brew
            # https://github.com/caffe2/tutorials/blob/master/MNIST.ipynb
            # fc = brew.fc(model, layer, tag_fc_w, dim_in=m, dim_out=n)
            layers.append(fc)

            if i == sigmoid_layer:
                # approach 1: construct sigmoid operator using model.net
                layer = self.AddLayerWrapper(model.net.Sigmoid, tag_fc_y, tag_fc_z)
                # approach 2: using brew (which currently does not support sigmoid)
                # tag_sigm = tag_layer + ":::" + "sigmoid" + str(i)
                # layer = brew.sigmoid(model,fc,tag_sigmoid)
            else:
                # approach 1: construct relu operator using model.net
                layer = self.AddLayerWrapper(model.net.Relu, tag_fc_y, tag_fc_z)
                # approach 2: using brew
                # tag_relu = tag_layer + ":::" + "relu" + str(i)
                # layer = brew.relu(model,fc,tag_relu)
            tag_in = tag_fc_z
            layers.append(layer)

        # WARNING: the dependency between layers is implicit in the tags,
        # so only the last layer is added to the layers list. It will
        # later be used for interactions.
        return layers, weights

    def create_emb(self, m, ln, model, tag):
        (tag_layer, tag_in, tag_out) = tag
        emb_l = []
        weights_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # select device
            if self.ndevices > 1:
                d = i % self.ndevices
            else:
                d = -1

            # create tags
            on_device = "" if self.ndevices <= 1 else "gpu_" + str(d) + "/"
            len_s = on_device + tag_layer + ":::" + "sls" + str(i) + "_l"
            ind_s = on_device + tag_layer + ":::" + "sls" + str(i) + "_i"
            tbl_s = on_device + tag_layer + ":::" + "sls" + str(i) + "_w"
            sum_s = on_device + tag_layer + ":::" + "sls" + str(i) + "_z"
            weights_l.append(tbl_s)

            # initialize the weights
            # approach 1a: custom
            W = np.random.uniform(low=-np.sqrt(1 / n),
                                  high=np.sqrt(1 / n),
                                  size=(n, m)).astype(np.float32)
            # approach 1b: numpy rand
            # W = ra.rand(n, m).astype(np.float32)
            self.FeedBlobWrapper(tbl_s, W, False, device_id=d)
            # approach 2: caffe2 xavier
            # with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, d)):
            #     W = model.param_init_net.XavierFill([], tbl_s, shape=[n, m])
            # save the blob shapes for latter (only needed if onnx is requested)
            if self.save_onnx:
                self.onnx_tsd[tbl_s] = (onnx.TensorProto.FLOAT, W.shape)

            # create operator
            if self.ndevices <= 1:
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s])
            else:
                with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, d)):
                    EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s])
            emb_l.append(EE)

        return emb_l, weights_l

    def create_interactions(self, x, ly, model, tag):
        (tag_dense_in, tag_sparse_in, tag_int_out) = tag

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            tag_int_out_info = tag_int_out + "_info"
            T, T_info = model.net.Concat(
                x + ly,
                [tag_int_out + "_cat_axis0", tag_int_out_info + "_cat_axis0"],
                axis=1,
                add_axis=1,
            )
            # perform a dot product
            Z = model.net.BatchMatMul([T, T], tag_int_out + "_matmul", trans_b=1)
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = model.net.Flatten(Z, tag_int_out + "_flatten", axis=1)
            # approach 2: unique
            Zflat_all = model.net.Flatten(Z, tag_int_out + "_flatten_all", axis=1)
            Zflat = model.net.BatchGather(
                [Zflat_all, tag_int_out + "_tril_indices"],
                tag_int_out + "_flatten"
            )
            R, R_info = model.net.Concat(
                x + [Zflat], [tag_int_out, tag_int_out_info], axis=1
            )
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            tag_int_out_info = tag_int_out + "_info"
            R, R_info = model.net.Concat(
                x + ly, [tag_int_out, tag_int_out_info], axis=1
            )
        else:
            sys.exit("ERROR: --arch-interaction-op="
                     + self.arch_interaction_op + " is not supported")

        return R

    def create_sequential_forward_ops(self):
        # embeddings
        tag = (self.temb, self.tsin, self.tsout)
        self.emb_l, self.emb_w = self.create_emb(self.m_spa, self.ln_emb,
                                                 self.model, tag)
        # bottom mlp
        tag = (self.tbot, self.tdin, self.tdout)
        self.bot_l, self.bot_w = self.create_mlp(self.ln_bot, self.sigmoid_bot,
                                                 self.model, tag)
        # interactions
        tag = (self.tdout, self.tsout, self.tint)
        Z = self.create_interactions([self.bot_l[-1]], self.emb_l, self.model, tag)

        # top mlp
        tag = (self.ttop, Z, self.tout)
        self.top_l, self.top_w = self.create_mlp(self.ln_top, self.sigmoid_top,
                                                 self.model, tag)
        # debug prints
        # print(self.emb_l)
        # print(self.bot_l)
        # print(self.top_l)

        # setup the last output variable
        self.last_output = self.top_l[-1]

    def create_parallel_forward_ops(self):
        # distribute embeddings (model parallelism)
        tag = (self.temb, self.tsin, self.tsout)
        self.emb_l, self.emb_w = self.create_emb(self.m_spa, self.ln_emb,
                                                 self.model, tag)
        # replicate mlp (data parallelism)
        tag = (self.tbot, self.tdin, self.tdout)
        self.bot_l, self.bot_w = self.create_mlp(self.ln_bot, self.sigmoid_bot,
                                                 self.model, tag)

        # add communication (butterfly shuffle)
        t_list = []
        for i, emb_output in enumerate(self.emb_l):
            # split input
            src_d = i % self.ndevices
            lo = [emb_output + "_split_" + str(d) for d in range(self.ndevices)]
            # approach 1: np and caffe2 operators assume the mini-batch size is
            # divisible exactly by the number of available devices
            with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, src_d)):
                self.model.net.Split(emb_output, lo, axis=0)
            """
            # approach 2: np and caffe2 operators do not assume exact divisibility
            ls = where_to_split(args.mini_batch_size, self.ndevices, _add_leftover=True)
            with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, src_d)):
                emb_output_split = self.model.net.Split(
                    emb_output, lo, split=lp, axis=0
                )
            """
            # scatter
            y = []
            for dst_d in range(len(lo)):
                src_blob = lo[dst_d]
                dst_blob = str(src_blob).replace(
                    "gpu_" + str(src_d), "gpu_" + str(dst_d), 1
                )
                if src_blob != dst_blob:
                    with core.DeviceScope(
                            core.DeviceOption(workspace.GpuDeviceType, dst_d)
                    ):
                        blob = self.model.Copy(src_blob, dst_blob)
                else:
                    blob = dst_blob
                y.append(blob)
            t_list.append(y)
        # adjust lists to be ordered per device
        x = list(map(lambda x: list(x), zip(*self.bot_l)))
        ly = list(map(lambda y: list(y), zip(*t_list)))

        # interactions
        for d in range(self.ndevices):
            on_device = "gpu_" + str(d) + "/"
            tag = (on_device + self.tdout, on_device + self.tsout, on_device + self.tint)
            with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, d)):
                self.create_interactions([x[d][-1]], ly[d], self.model, tag)

        # replicate mlp (data parallelism)
        tag = (self.ttop, self.tint, self.tout)
        self.top_l, self.top_w = self.create_mlp(self.ln_top, self.sigmoid_top,
                                                 self.model, tag)

        # debug prints
        # print(self.model.net.Proto(),end='\n')
        # sys.exit("ERROR: debugging")

        # setup the last output variable
        self.last_output = self.top_l[-1]

    def __init__(
        self,
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        save_onnx=False,
        model=None,
        test_net=None,
        tag=None,
        ndevices=-1,
        forward_ops=True,
        enable_prof=False,
    ):
        super(DLRM_Net, self).__init__()

        # init model
        if model is None:
            global_init_opt = ["caffe2", "--caffe2_log_level=0"]
            if enable_prof:
                global_init_opt += [
                    "--logtostderr=0",
                    "--log_dir=$HOME",
                    "--caffe2_logging_print_net_summary=1",
                ]
            workspace.GlobalInit(global_init_opt)
            self.set_tags()
            self.model = model_helper.ModelHelper(name="DLRM", init_params=True)
            self.test_net = None
        else:
            # WARNING: assume that workspace and tags have been initialized elsewhere
            self.set_tags(tag[0], tag[1], tag[2], tag[3], tag[4], tag[5], tag[6],
                          tag[7], tag[8], tag[9])
            self.model = model
            self.test_net = test_net

        # save arguments
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top
        self.arch_interaction_op = arch_interaction_op
        self.arch_interaction_itself = arch_interaction_itself
        self.sigmoid_bot = sigmoid_bot
        self.sigmoid_top = sigmoid_top
        self.save_onnx = save_onnx
        self.ndevices = ndevices
        # onnx types and shapes dictionary
        if self.save_onnx:
            self.onnx_tsd = {}
        # create forward operators
        if forward_ops:
            if self.ndevices <= 1:
                return self.create_sequential_forward_ops()
            else:
                return self.create_parallel_forward_ops()

    def set_tags(
        self,
        _tag_layer_top_mlp="top",
        _tag_layer_bot_mlp="bot",
        _tag_layer_embedding="emb",
        _tag_feature_dense_in="dense_in",
        _tag_feature_dense_out="dense_out",
        _tag_feature_sparse_in="sparse_in",
        _tag_feature_sparse_out="sparse_out",
        _tag_interaction="interaction",
        _tag_dense_output="prob_click",
        _tag_dense_target="target",
    ):
        # layer tags
        self.ttop = _tag_layer_top_mlp
        self.tbot = _tag_layer_bot_mlp
        self.temb = _tag_layer_embedding
        # dense feature tags
        self.tdin = _tag_feature_dense_in
        self.tdout = _tag_feature_dense_out
        # sparse feature tags
        self.tsin = _tag_feature_sparse_in
        self.tsout = _tag_feature_sparse_out
        # output and target tags
        self.tint = _tag_interaction
        self.ttar = _tag_dense_target
        self.tout = _tag_dense_output

    def parameters(self):
        return self.model

    def get_loss(self):
        return self.FetchBlobWrapper(self.loss, reduce_across="add")

    def get_output(self):
        return self.FetchBlobWrapper(self.last_output, reduce_across="concat")

    def create(self, X, S_lengths, S_indices, T):
        self.create_input(X, S_lengths, S_indices, T)
        self.create_model(X, S_lengths, S_indices, T)

    def create_input(self, X, S_lengths, S_indices, T):
        # feed input data to blobs
        self.FeedBlobWrapper(self.tdin, X, split=True)
        # save the blob shapes for latter (only needed if onnx is requested)
        if self.save_onnx:
            self.onnx_tsd[self.tdin] = (onnx.TensorProto.FLOAT, X.shape)

        for i in range(len(self.emb_l)):
            # select device
            if self.ndevices > 1:
                d = i % self.ndevices
            else:
                d = -1
            # create tags
            on_device = "" if self.ndevices <= 1 else "gpu_" + str(d) + "/"
            len_s = on_device + self.temb + ":::" + "sls" + str(i) + "_l"
            ind_s = on_device + self.temb + ":::" + "sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]), False, device_id=d)
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]), False, device_id=d)
            # save the blob shapes for latter (only needed if onnx is requested)
            if self.save_onnx:
                lshape = (len(S_lengths[i]),)  # =args.mini_batch_size
                ishape = (len(S_indices[i]),)
                self.onnx_tsd[len_s] = (onnx.TensorProto.INT32, lshape)
                self.onnx_tsd[ind_s] = (onnx.TensorProto.INT32, ishape)

        # feed target data to blobs
        if T is not None:
            zeros_fp32 = np.zeros(T.shape).astype(np.float32)
            self.FeedBlobWrapper(self.ttar, zeros_fp32, split=True)
            # save the blob shapes for latter (only needed if onnx is requested)
            if self.save_onnx:
                self.onnx_tsd[self.ttar] = (onnx.TensorProto.FLOAT, T.shape)

    def create_model(self, X, S_lengths, S_indices, T):
        #setup tril indices for the interactions
        offset = 1 if self.arch_interaction_itself else 0
        num_fea = len(self.emb_l) + 1
        tril_indices = np.array([j + i * num_fea
                                 for i in range(num_fea) for j in range(i + offset)])
        self.FeedBlobWrapper(self.tint + "_tril_indices", tril_indices)
        if self.save_onnx:
            tish = tril_indices.shape
            self.onnx_tsd[self.tint + "_tril_indices"] = (onnx.TensorProto.INT32, tish)

        # create compute graph
        if T is not None:
            # WARNING: RunNetOnce call is needed only if we use brew and ConstantFill.
            # We could use direct calls to self.model functions above to avoid it
            workspace.RunNetOnce(self.model.param_init_net)
            workspace.CreateNet(self.model.net)
            if self.test_net is not None:
                workspace.CreateNet(self.test_net)

    def run(self, X, S_lengths, S_indices, T, test_net=False, enable_prof=False):
        # feed input data to blobs
        # dense features
        self.FeedBlobWrapper(self.tdin, X, split=True)
        # sparse features
        for i in range(len(self.emb_l)):
            # select device
            if self.ndevices > 1:
                d = i % self.ndevices
            else:
                d = -1
            # create tags
            on_device = "" if self.ndevices <= 1 else "gpu_" + str(d) + "/"
            len_s = on_device + self.temb + ":::" + "sls" + str(i) + "_l"
            ind_s = on_device + self.temb + ":::" + "sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]), False, device_id=d)
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]), False, device_id=d)

        # feed target data to blobs if needed
        if T is not None:
            self.FeedBlobWrapper(self.ttar, T, split=True)
            # execute compute graph
            if test_net:
                workspace.RunNet(self.test_net)
            else:
                if enable_prof:
                    workspace.C.benchmark_net(self.model.net.Name(), 0, 1, True)
                else:
                    workspace.RunNet(self.model.net)
        # debug prints
        # print("intermediate")
        # print(self.FetchBlobWrapper(self.bot_l[-1]))
        # for tag_emb in self.emb_l:
        #     print(self.FetchBlobWrapper(tag_emb))
        # print(self.FetchBlobWrapper(self.tint))

    def MSEloss(self, scale=1.0):
        # add MSEloss to the model
        self.AddLayerWrapper(self.model.SquaredL2Distance, [self.tout, self.ttar], "sd")
        self.AddLayerWrapper(self.model.Scale, "sd", "sd2", scale=2.0 * scale)
        # WARNING: "loss" is a special tag and should not be changed
        self.loss = self.AddLayerWrapper(self.model.AveragedLoss, "sd2", "loss")

    def BCEloss(self, scale=1.0, threshold=0.0):
        # add BCEloss to the mode
        if 0.0 < threshold and threshold < 1.0:
            self.AddLayerWrapper(self.model.Clip, self.tout, "tout_c",
                                 min=threshold, max=(1.0 - threshold))
            self.AddLayerWrapper(self.model.MakeTwoClass, "tout_c", "tout_2c")
        else:
            self.AddLayerWrapper(self.model.MakeTwoClass, self.tout, "tout_2c")
        self.AddLayerWrapper(self.model.LabelCrossEntropy, ["tout_2c", self.ttar], "sd")
        # WARNING: "loss" is a special tag and should not be changed
        if scale == 1.0:
            self.loss = self.AddLayerWrapper(self.model.AveragedLoss, "sd", "loss")
        else:
            self.AddLayerWrapper(self.model.Scale, "sd", "sd2", scale=scale)
            self.loss = self.AddLayerWrapper(self.model.AveragedLoss, "sd2", "loss")

    def sgd_optimizer(self, learning_rate,
                      T=None, _gradientMap=None, sync_dense_params=True):
        # create one, it and lr tags (or use them if already present)
        if T is not None:
            (tag_one, tag_it, tag_lr) = T
        else:
            (tag_one, tag_it, tag_lr) = ("const_one", "optim_it", "optim_lr")

            # approach 1: feed values directly
            # self.FeedBlobWrapper(tag_one, np.ones(1).astype(np.float32))
            # self.FeedBlobWrapper(tag_it, np.zeros(1).astype(np.int64))
            # it = self.AddLayerWrapper(self.model.Iter, tag_it, tag_it)
            # lr = self.AddLayerWrapper(self.model.LearningRate, tag_it, tag_lr,
            #                           base_lr=-1 * learning_rate, policy="fixed")
            # approach 2: use brew
            self.AddLayerWrapper(self.model.param_init_net.ConstantFill,
                                 [], tag_one, shape=[1], value=1.0)
            self.AddLayerWrapper(brew.iter, self.model, tag_it)
            self.AddLayerWrapper(self.model.LearningRate, tag_it, tag_lr,
                                 base_lr=-1 * learning_rate, policy="fixed")
            # save the blob shapes for latter (only needed if onnx is requested)
            if self.save_onnx:
                self.onnx_tsd[tag_one] = (onnx.TensorProto.FLOAT, (1,))
                self.onnx_tsd[tag_it] = (onnx.TensorProto.INT64, (1,))

        # create gradient maps (or use them if already present)
        if _gradientMap is not None:
            self.gradientMap = _gradientMap
        else:
            if self.loss.__class__ == list:
                self.gradientMap = self.model.AddGradientOperators(self.loss)
            else:
                self.gradientMap = self.model.AddGradientOperators([self.loss])

        # update weights
        # approach 1: builtin function
        # optimizer.build_sgd(self.model, base_learning_rate=learning_rate)
        # approach 2: custom code
        # top MLP weight and bias
        for w in self.top_w:
            # allreduce across devices if needed
            if sync_dense_params and self.ndevices > 1:
                grad_blobs = [
                    self.gradientMap["gpu_{}/".format(d) + w]
                    for d in range(self.ndevices)
                ]
                self.model.NCCLAllreduce(grad_blobs, grad_blobs)
            # update weights
            self.AddLayerWrapper(self.model.WeightedSum,
                                 [w, tag_one, "", tag_lr], w, reset_grad=True)
        # bottom MLP weight and bias
        for w in self.bot_w:
            # allreduce across devices if needed
            if sync_dense_params and self.ndevices > 1:
                grad_blobs = [
                    self.gradientMap["gpu_{}/".format(d) + w]
                    for d in range(self.ndevices)
                ]
                self.model.NCCLAllreduce(grad_blobs, grad_blobs)
            # update weights
            self.AddLayerWrapper(self.model.WeightedSum,
                                 [w, tag_one, "", tag_lr], w, reset_grad=True)
        # update embeddings
        for i, w in enumerate(self.emb_w):
            # select device
            if self.ndevices > 1:
                d = i % self.ndevices
            # create tags
            on_device = "" if self.ndevices <= 1 else "gpu_" + str(d) + "/"
            _tag_one = on_device + tag_one
            _tag_lr = on_device + tag_lr
            # pickup gradient
            w_grad = self.gradientMap[w]
            # update weights
            if self.ndevices > 1:
                with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, d)):
                    self.model.ScatterWeightedSum([w, _tag_one, w_grad.indices,
                                                   w_grad.values, _tag_lr], w)
            else:
                self.model.ScatterWeightedSum([w, _tag_one, w_grad.indices,
                                               w_grad.values, _tag_lr], w)

    def print_all(self):
        # approach 1: all
        print(workspace.Blobs(), end='\n')
        for _, l in enumerate(workspace.Blobs()):
            print(l)
            print(self.FetchBlobWrapper(l))
        # approach 2: only summary
        # for param in self.model.params:
        #    self.model.Summarize(param, [], to_file=1)
        #    self.model.Summarize(self.model.param_to_grad[param], [], to_file=1)

    def print_weights(self):
        for _, l in enumerate(self.emb_w):
            # print(l)
            print(self.FetchBlobWrapper(l, False))
        for _, l in enumerate(self.bot_w):
            # print(l)
            if self.ndevices > 1:
                print(self.FetchBlobWrapper(l, False, device_id=0))
            else:
                print(self.FetchBlobWrapper(l))
        for _, l in enumerate(self.top_w):
            # print(l)
            if self.ndevices > 1:
                print(self.FetchBlobWrapper(l, False, device_id=0))
            else:
                print(self.FetchBlobWrapper(l))

    def print_activations(self):
        for _, l in enumerate(self.emb_l):
            print(l)
            print(self.FetchBlobWrapper(l, False))
        for _, l in enumerate(self.bot_l):
            print(l)
            print(self.FetchBlobWrapper(l))
        print(self.tint)
        print(self.FetchBlobWrapper(self.tint))
        for _, l in enumerate(self.top_l):
            print(l)
            print(self.FetchBlobWrapper(l))


def define_metrics():
    metrics = {
        'loss': lambda y_true, y_score:
        sklearn.metrics.log_loss(
            y_true=y_true,
            y_pred=y_score,
            labels=[0,1]),
        'recall': lambda y_true, y_score:
        sklearn.metrics.recall_score(
            y_true=y_true,
            y_pred=np.round(y_score)
        ),
        'precision': lambda y_true, y_score:
        sklearn.metrics.precision_score(
            y_true=y_true,
            y_pred=np.round(y_score)
        ),
        'f1': lambda y_true, y_score:
        sklearn.metrics.f1_score(
            y_true=y_true,
            y_pred=np.round(y_score)
        ),
        'ap': sklearn.metrics.average_precision_score,
        'roc_auc': sklearn.metrics.roc_auc_score,
        'accuracy': lambda y_true, y_score:
        sklearn.metrics.accuracy_score(
            y_true=y_true,
            y_pred=np.round(y_score)
        ),
        # 'pre_curve' : sklearn.metrics.precision_recall_curve,
        # 'roc_curve' :  sklearn.metrics.roc_curve,
    }
    return metrics


def calculate_metrics(targets, scores):
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    metrics = define_metrics()

    # print("Compute time for validation metric : ", end="")
    # first_it = True
    validation_results = {}
    for metric_name, metric_function in metrics.items():
        # if first_it:
        #     first_it = False
        # else:
        #     print(", ", end="")
        # metric_compute_start = time_wrap(False)
        try:
            validation_results[metric_name] = metric_function(
                targets,
                scores
            )
        except Exception as error :
            validation_results[metric_name] = -1
            print("{} in calculating {}".format(error, metric_name))
        # metric_compute_end = time_wrap(False)
        # met_time = metric_compute_end - metric_compute_start
        # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
        #      end="")
    # print(" ms")
    return validation_results

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
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")   # or bce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-generation", type=str, default="random")  # or synthetic or dataset
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
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--caffe2-net-type", type=str, default="")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx (or protobuf with shapes)
    parser.add_argument("--save-onnx", action="store_true", default=False)
    parser.add_argument("--save-proto-types-shapes", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    use_gpu = args.use_gpu
    if use_gpu:
        device_opt = core.DeviceOption(workspace.GpuDeviceType, 0)
        ngpus = workspace.NumGpuDevices()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device_opt = core.DeviceOption(caffe2_pb2.CPU)
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    if args.data_generation == "dataset":
        # input and target from dataset
        (nbatches, lX, lS_l, lS_i, lT,
         nbatches_test, lX_test, lS_l_test, lS_i_test, lT_test,
         ln_emb, m_den) = dc.read_dataset(
             args.data_set, args.max_ind_range, args.data_sub_sample_rate,
             args.mini_batch_size, args.num_batches, args.data_randomize, "train",
             args.raw_data_file, args.processed_data_file, args.memory_map
        )
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        (nbatches, lX, lS_l, lS_i, lT) = dc.generate_random_data(
            m_den, ln_emb, args.data_size, args.num_batches, args.mini_batch_size,
            args.num_indices_per_lookup, args.num_indices_per_lookup_fixed,
            1, args.round_targets, args.data_generation, args.data_trace_file,
            args.data_trace_enable_padding
        )

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
        sys.exit("ERROR: --arch-interaction-op="
                 + args.arch_interaction_op + " is not supported")
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit("ERROR: arch-dense-feature-size "
            + str(m_den) + " does not match first dim of bottom mlp " + str(ln_bot[0]))
    if m_spa != m_den_out:
        sys.exit("ERROR: arch-sparse-feature-size "
            + str(m_spa) + " does not match last dim of bottom mlp " + str(m_den_out))
    if num_int != ln_top[0]:
        sys.exit("ERROR: # of feature interactions "
            + str(num_int) + " does not match first dim of top mlp " + str(ln_top[0]))

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print("mlp top arch " + str(ln_top.size - 1)
              + " layers, with input to output dimensions:")
        print(ln_top)

        print("# of interactions")
        print(num_int)
        print("mlp bot arch " + str(ln_bot.size - 1)
              + " layers, with input to output dimensions:")
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print("# of embeddings (= # of sparse features) " + str(ln_emb.size)
              + ", with dimensions " + str(m_spa) + "x:")
        print(ln_emb)

        print("data (inputs and targets):")
        for j in range(0, nbatches):
            print("mini-batch: %d" % j)
            print(lX[j])
            print(lS_l[j])
            print(lS_i[j])
            print(lT[j].astype(np.float32))

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1
    flag_types_shapes = args.save_onnx or args.save_proto_types_shapes
    flag_forward_ops = not (use_gpu and ndevices > 1)
    with core.DeviceScope(device_opt):
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 1,
            save_onnx=flag_types_shapes,
            ndevices=ndevices,
            # forward_ops = flag_forward_ops
            enable_prof=args.enable_profiling,
        )
    # load nccl if using multiple devices
    if args.sync_dense_params and ndevices > 1:
        dyndep.InitOpsLibrary("//caffe2/caffe2/contrib/nccl:nccl_ops")
    # set the net type for better performance (dag, async_scheduling, etc)
    if args.caffe2_net_type:
        dlrm.parameters().net.Proto().type = args.caffe2_net_type
    # plot compute graph
    if args.plot_compute_graph:
        graph = net_drawer.GetPydotGraph(
            dlrm.parameters().net,
            "dlrm_s_caffe2_graph",
            "BT"
        )
        graph.write_pdf(graph.get_name() + ".pdf")
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        dlrm.print_weights()

    # add training loss if needed
    if not args.inference_only:
        with core.DeviceScope(device_opt):
            # specify the loss function
            nd = 1.0 if dlrm.ndevices <= 1 else 1.0 / dlrm.ndevices  # 1
            if args.loss_function == "mse":
                dlrm.MSEloss(scale=nd)
            elif args.loss_function == "bce":
                dlrm.BCEloss(scale=nd, threshold=args.loss_threshold)
            else:
                sys.exit("ERROR: --loss-function=" + args.loss_function
                         + " is not supported")

            # define test net (as train net without gradients)
            dlrm.test_net = core.Net(copy.deepcopy(dlrm.model.net.Proto()))

            # specify the optimizer algorithm
            dlrm.sgd_optimizer(
                args.learning_rate, sync_dense_params=args.sync_dense_params
            )
    # init/create
    dlrm.create(lX[0], lS_l[0], lS_i[0], lT[0])

    ### main loop ###
    best_gA_test = 0
    best_auc_test = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0

    print("time/loss/accuracy (if enabled):")
    while k < args.nepochs:
        j = 0
        while j < nbatches:
            '''
            # debug prints
            print("input and targets")
            print(lX[j])
            print(lS_l[j])
            print(lS_i[j])
            print(lT[j].astype(np.float32))
            '''
            # forward and backward pass, where the latter runs only
            # when gradients and loss have been added to the net
            time1 = time.time()
            dlrm.run(lX[j], lS_l[j], lS_i[j], lT[j])  # args.enable_profiling
            time2 = time.time()
            total_time += time2 - time1

            # compte loss and accuracy
            Z = dlrm.get_output()  # numpy array
            T = lT[j]              # numpy array
            '''
            # debug prints
            print("output and loss")
            print(Z)
            print(dlrm.get_loss())
            '''
            mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
            A = np.sum((np.round(Z, 0) == T).astype(np.uint8))
            total_accu += 0 if args.inference_only else A
            total_loss += 0 if args.inference_only else dlrm.get_loss() * mbs
            total_iter += 1
            total_samp += mbs

            # print time, loss and accuracy
            should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
            should_test = (
                (args.test_freq > 0)
                and (args.data_generation == "dataset")
                and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
            )
            if should_print or should_test:
                gT = 1000. * total_time / total_iter if args.print_time else -1
                total_time = 0

                gA = total_accu / total_samp
                total_accu = 0

                gL = total_loss / total_samp
                total_loss = 0

                str_run_type = "inference" if args.inference_only else "training"
                print(
                    "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                        str_run_type, j + 1, nbatches, k, gT
                    )
                    + " loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                )
                total_iter = 0
                total_samp = 0
                # debug prints
                # print(Z)
                # print(T)

                # testing
                if should_test and not args.inference_only:
                    # don't measure training iter time in a test iteration
                    if args.mlperf_logging:
                        previous_iteration_time = None

                    test_accu = 0
                    test_loss = 0
                    test_samp = 0

                    if args.mlperf_logging:
                        scores = []
                        targets = []

                    for i in range(nbatches_test):
                        # early exit if nbatches was set by the user and was exceeded
                        if nbatches > 0 and i >= nbatches:
                            break

                        # forward pass
                        dlrm.run(lX_test[i], lS_l_test[i], lS_i_test[i], lT_test[i], test_net=True)
                        Z_test = dlrm.get_output()
                        T_test = lT_test[i]

                        if args.mlperf_logging:
                            scores.append(Z_test)
                            targets.append(T_test)
                        else:
                            # compte loss and accuracy
                            L_test = dlrm.get_loss()
                            mbs_test = T_test.shape[0]  # = mini_batch_size except last
                            A_test = np.sum((np.round(Z_test, 0) == T_test).astype(np.uint8))
                            test_accu += A_test
                            test_loss += L_test * mbs_test
                            test_samp += mbs_test

                    # compute metrics (after test loop has finished)
                    if args.mlperf_logging:
                        validation_results = calculate_metrics(targets, scores)
                        gA_test = validation_results['accuracy']
                        gL_test = validation_results['loss']
                    else:
                        gA_test = test_accu / test_samp
                        gL_test = test_loss / test_samp

                    # print metrics
                    is_best = gA_test > best_gA_test
                    if is_best:
                        best_gA_test = gA_test

                    if args.mlperf_logging:
                        is_best = validation_results['roc_auc'] > best_auc_test
                        if is_best:
                            best_auc_test = validation_results['roc_auc']

                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                            + " loss {:.6f}, recall {:.4f}, precision {:.4f},".format(
                                validation_results['loss'],
                                validation_results['recall'],
                                validation_results['precision']
                            )
                            + " f1 {:.4f}, ap {:.4f},".format(
                                validation_results['f1'],
                                validation_results['ap'],
                            )
                            + " auc {:.4f}, best auc {:.4f},".format(
                                validation_results['roc_auc'],
                                best_auc_test
                            )
                            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                                validation_results['accuracy'] * 100,
                                best_gA_test * 100
                            )
                        )
                    else:
                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 0)
                            + " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                gL_test, gA_test * 100, best_gA_test * 100
                            )
                        )

                    # check thresholds
                    if (args.mlperf_logging
                        and (args.mlperf_acc_threshold > 0)
                        and (best_gA_test > args.mlperf_acc_threshold)):
                        print("MLPerf testing accuracy threshold "
                              + str(args.mlperf_acc_threshold)
                              + " reached, stop training")
                        break

                    if (args.mlperf_logging
                        and (args.mlperf_auc_threshold > 0)
                        and (best_auc_test > args.mlperf_auc_threshold)):
                        print("MLPerf testing auc threshold "
                              + str(args.mlperf_auc_threshold)
                              + " reached, stop training")
                        break


            j += 1 # nbatches
        k += 1  # nepochs

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        dlrm.print_weights()

    # build onnx model from caffe2
    if args.save_onnx:
        pnet = dlrm.parameters().net.Proto()
        inet = dlrm.parameters().param_init_net.Proto()
        value_info = dlrm.onnx_tsd  # None
        # debug prints
        # print(value_info)

        # WARNING: Why Caffe2 to ONNX net transformation currently does not work?
        # ONNX does not support SparseLengthsSum operator directly. A workaround
        # could be for the Caffe2 ONNX frontend to indirectly map this operator to
        # Gather and ReducedSum ONNX operators, following the PyTorch approach.
        c2f = caffe2.python.onnx.frontend.Caffe2Frontend()
        dlrm_caffe2_onnx = c2f.caffe2_net_to_onnx_model(pnet, inet, value_info)
        # check the onnx model
        onnx.checker.check_model(dlrm_caffe2_onnx)

        # save model to a file
        with open("dlrm_s_caffe2.onnx", "w+") as dlrm_caffe2_onnx_file:
            dlrm_caffe2_onnx_file.write(str(dlrm_caffe2_onnx))

    # build protobuf with types and shapes
    if args.save_proto_types_shapes:
        # add types and shapes to protobuf
        __TYPE_MAPPING = {
            onnx.TensorProto.FLOAT: caffe2_pb2.TensorProto.FLOAT,
            onnx.TensorProto.UINT8: caffe2_pb2.TensorProto.UINT8,
            onnx.TensorProto.INT8: caffe2_pb2.TensorProto.INT8,
            onnx.TensorProto.UINT16: caffe2_pb2.TensorProto.UINT16,
            onnx.TensorProto.INT16: caffe2_pb2.TensorProto.INT16,
            onnx.TensorProto.INT32: caffe2_pb2.TensorProto.INT32,
            onnx.TensorProto.INT64: caffe2_pb2.TensorProto.INT64,
            onnx.TensorProto.STRING: caffe2_pb2.TensorProto.STRING,
            onnx.TensorProto.BOOL: caffe2_pb2.TensorProto.BOOL,
            onnx.TensorProto.FLOAT16: caffe2_pb2.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE: caffe2_pb2.TensorProto.DOUBLE,
        }

        pnet = dlrm.parameters().net.Proto()
        arg = pnet.arg.add()
        arg.name = "input_shape_info"
        for i in pnet.external_input:
            if i in dlrm.onnx_tsd:
                onnx_dtype, shape = dlrm.onnx_tsd[i]
                t = arg.tensors.add()
                t.name = i
                t.data_type = __TYPE_MAPPING[onnx_dtype]
                t.dims.extend(shape)
            else:
                print("Warning: we don't have shape/type info for input: {}".format(i))
        # debug print
        # print(pnet)

        # export the protobuf with types and shapes
        with open("dlrm_s_caffe2.proto", "w+") as dlrm_s_proto_file:
            dlrm_s_proto_file.write(str(pnet))

        """
        # export the protobuf with types and shapes as well as weights
        # see https://github.com/pytorch/pytorch/issues/9533
        #save
        net = dlrm.parameters().net
        params = dlrm.parameters().params
        init_net, predict_net = mobile_exporter.Export(workspace, net, params)
        with open("dlrm_s_caffe2.predict", "wb") as dlrm_s_predict_file:
            dlrm_s_predict_file.write(predict_net.SerializeToString())
        with open("dlrm_s_caffe2.init", "wb") as dlrm_s_init_file:
            dlrm_s_init_file.write(init_net.SerializeToString())
        #load
        net_def = caffe2_pb2.NetDef()
        init_def= caffe2_pb2.NetDef()
        with open("dlrm_s_caffe2.predict", "rb") as dlrm_s_predict_file:
            net_def.ParseFromString(dlrm_s_predict_file.read())
            print(net_def)
        with open("dlrm_s_caffe2.init", "rb") as dlrm_s_init_file:
            init_def.ParseFromString(dlrm_s_init_file.read())
            print(init_def)
        """
