# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import functools
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.autograd import Function

import sys
sys.path.append("../..")
import graph

object_id = 0

class TensorWrapper(object):
    def __init__(self, tensor, node_desc, graph_creator, activation_size=None):
        self.tensor = tensor
        self.shape = tensor.shape
        global object_id
        self.object_id = object_id
        object_id += 1
        self.node_desc = node_desc

        i = 0
        for i in range(len(graph_creator.summary)):
            if str(graph_creator.summary[i]['layer_name']) == node_desc:
                break

        if i < len(graph_creator.summary) and node_desc == str(graph_creator.summary[i]['layer_name']):
            summary_elem = graph_creator.summary.pop(i)
            forward_compute_time = summary_elem['forward_time']
            backward_compute_time = summary_elem['backward_time']
            if isinstance(summary_elem['output_shape'][0], list):
                activation_sizes = [4.0 * functools.reduce(lambda x, y: x * y, elem)
                                    for elem in summary_elem['output_shape']]
            else:
                activation_sizes = 4.0 * functools.reduce(lambda x, y: x * y, summary_elem['output_shape'])
            parameter_size = 4.0 * float(summary_elem['nb_params'])
            self._node = graph.Node("node%d" % object_id, node_desc=node_desc,
                                    forward_compute_time=forward_compute_time,
                                    backward_compute_time=backward_compute_time,
                                    activation_size=activation_sizes,
                                    parameter_size=parameter_size)
        elif activation_size is not None:
            self._node = graph.Node("node%d" % object_id, node_desc=node_desc,
                                    activation_size=activation_size)
        else:
            self._node = graph.Node("node%d" % object_id, node_desc=node_desc)
        self.graph_creator = graph_creator

    def size(self, dim=None):
        if dim is None:
            result = self.tensor.size()
            dim_str = ""
        else:
            result = self.tensor.size(dim)
            dim_str = "(%d)" % dim
        wrapped_result = TensorWrapper(result, "Size%s" % dim_str, self.graph_creator,
                                       activation_size=4)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def dim(self):
        return self.tensor.dim()

    def view(self, *wrapped_sizes):
        sizes = []
        in_edges = []
        for wrapped_size in wrapped_sizes:
            if isinstance(wrapped_size, TensorWrapper):
                sizes.append(wrapped_size.tensor)
                in_edges.append(wrapped_size)
            else:
                sizes.append(wrapped_size)
        result = self.tensor.view(*sizes)
        if len(sizes) == 1:
            wrapped_result = TensorWrapper(result, "View", self.graph_creator,
                                           activation_size=self.node().activation_size)
        else:
            wrapped_result = TensorWrapper(result,
                                           "View(%s)" % ", ".join([str(size) for size in sizes[1:]]),
                                           self.graph_creator,
                                           activation_size=self.node().activation_size)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        for in_edge in in_edges:
            self.graph_creator.graph.add_edge(in_edge.node(), wrapped_result.node())
        return wrapped_result

    def __gt__(self, other):
        return self.tensor.__gt__(other)

    def __lt__(self, other):
        return self.tensor.__lt__(other)

    def __add__(self, other):
        self_activation_size = self.node().activation_size
        if isinstance(other, TensorWrapper):
            other_activation_size = other.node().activation_size
            assert(self_activation_size == other_activation_size)
            result_tensor = self.tensor + other.tensor
        else:
            result_tensor = self.tensor + other
        wrapped_result = TensorWrapper(result_tensor, "Add", self.graph_creator,
                                       activation_size=self_activation_size)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        if isinstance(other, TensorWrapper):
            self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __iadd__(self, other):
        self_activation_size = self.node().activation_size
        other_activation_size = other.node().activation_size
        assert(self_activation_size == other_activation_size)
        wrapped_result = TensorWrapper(self.tensor, "Add(inplace)", self.graph_creator,
                                       activation_size=self_activation_size)
        self.tensor += other.tensor
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __mul__(self, other):
        result = self.tensor * other.tensor
        wrapped_result = TensorWrapper(result, "Mul", self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __getitem__(self, key):
        result_tensor = self.tensor[key]
        try:
            activation_size = self.node().activation_size[key]
        except:
            activation_size = self.node().activation_size
        wrapped_result = TensorWrapper(result_tensor, "__getitem__(%s)" % str(key), self.graph_creator,
                                       activation_size=activation_size)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def transpose(self, *args):
        result_tensor = self.tensor.transpose(*args)
        args_str = ", ".join([str(arg) for arg in args])
        wrapped_result = TensorWrapper(result_tensor, "Transpose(%s)" % args_str,
                                       self.graph_creator,
                                       activation_size=self.node().activation_size)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def unsqueeze(self, *args):
        return self.tensor.unsqueeze(*args)

    def node(self):
        return self._node

def bmm(a, b):
    if isinstance(a, TensorWrapper) and isinstance(b, TensorWrapper):
        result = torch.bmm(a.tensor, b.tensor)
        wrapped_result = TensorWrapper(result, "Bmm", a.graph_creator)
        a.graph_creator.graph.add_edge(a.node(), wrapped_result.node())
        a.graph_creator.graph.add_edge(b.node(), wrapped_result.node())
        return wrapped_result
    else:
        return torch.bmm(a, b)

    
def cat(wrapped_tensors, dim):
    tensors = []
    activation_sizes = []
    all_unwrapped_tensors = True
    graph_creator = None
    for wrapped_tensor in wrapped_tensors:
        if isinstance(wrapped_tensor, TensorWrapper):
            tensors.append(wrapped_tensor.tensor)
            activation_sizes.append(wrapped_tensor.node().activation_size)
            graph_creator = wrapped_tensor.graph_creator
            all_unwrapped_tensors = False
        else:
            tensors.append(wrapped_tensor)
    # Simplifying assumption: if all tensors are "unwrapped", then we're not profiling,
    # and default to torch implementation.
    if all_unwrapped_tensors:
        return torch.cat(tensors, dim)
    result = torch.cat(tensors, dim)
    wrapped_result = TensorWrapper(result, "Concat(%d)" % dim, graph_creator,
                                   activation_size=activation_sizes[0])
    for wrapped_tensor in wrapped_tensors:
        if not isinstance(wrapped_tensor, TensorWrapper):
            wrapped_tensor = TensorWrapper(wrapped_tensor, "Input", graph_creator)
        graph_creator.graph.add_edge(wrapped_tensor.node(), wrapped_result.node())
    return wrapped_result


class GraphCreator(object):
    def __init__(self, model, summary, module_whitelist):
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.summary = copy.deepcopy(summary)
        self.forward_original_methods = {}
        self.graph = graph.Graph()
        self.inputs = {}

    def hook_modules(self, module, root=False):
        this_creator = self
        sub_modules = module.__dict__['_modules']

        # Wrapper function to "forward()", keeping track of dependencies.
        def forward_wrapper(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            for i in range(len(wrapped_inputs_list)):
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    key = wrapped_inputs_list[i]
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]
                    else:
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        this_creator.inputs[key] = wrapped_inputs_list[i]
                    input.append(wrapped_inputs_list[i].tensor)
            result = this_creator.forward_original_methods[self](*input)
            wrapped_result = TensorWrapper(result, str(self), this_creator)
            for wrapped_input in wrapped_inputs_list:
                this_creator.graph.add_edge(wrapped_input.node(), wrapped_result.node())

            return wrapped_result

        # Wrapper function to "forward()", keeping track of dependencies.
        def forward_wrapper_root(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            for i in range(len(wrapped_inputs_list)):
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    key = wrapped_inputs_list[i]
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]
                    else:
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        this_creator.inputs[key] = wrapped_inputs_list[i]
                    input.append(wrapped_inputs_list[i].tensor)
            result = this_creator.forward_original_methods[self](*input)

            return result

        for name, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) == 0 or sub_module_name in self.module_whitelist:
                sub_module.reset_hooks()
                #
                # Hook nn.Module with no descendants.
                #

                # Replace "forward" with "wrapped_forward".
                if sub_module not in this_creator.forward_original_methods:
                    this_creator.forward_original_methods.update({sub_module:
                                                                   sub_module.forward})
                    sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants.
                #
                self.hook_modules(sub_module)
        if root:
            this_creator.forward_original_methods.update({module: module.forward})
            module.forward = forward_wrapper_root.__get__(module, module.__class__)

    def unhook_modules(self):
        for sub_module in self.forward_original_methods:
            sub_module.forward = self.forward_original_methods[sub_module]

    def persist_graph(self, directory):
        self.graph.to_dot(os.path.join(directory, "graph.dot"))
        with open(os.path.join(directory, "graph.txt"), 'w') as f:
            f.write(str(self.graph))
        self.graph.render_bar_graphs_and_cdfs(directory)
