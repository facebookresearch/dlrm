# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import re
import subprocess
import sys

sys.path.append("..")
import graph


declaration_whitelist = [
    "hidden",
    "__getitem__",
    "Add",
    "Mul",
    "Concat",
    "Input",
    "Size",
    "View",
    "Transpose",
    "self.get_seq_lens",
    "Bmm"
]

declaration_specialcase = [
    "EmuBidirLSTM",
    "RecurrentAttention",
    "Classifier",
    "MaskConv",
    "ResizeInput",
    "InferenceBatchSoftmax",
    "BatchRNN",
    "SequenceWise",
    "EmbeddingBag"
]

def get_output_tuple_str(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return "(%s)" % ", ".join(outputs)

def get_tensor_names_list(names):
    return [names[node_id] for node_id in sorted(names.keys())]

def get_input_names(graph, full_graph, check_stages=True):
    # Figure out the inputs to this sub-graph, which are the predecessors of
    # nodes in the sub-graph not in the sub-graph.
    # input_names is a dict mapping each predecessor's node_id to assigned
    # variable name.
    nodes = graph.nodes
    input_names = {}
    counter = 0
    for node_id in nodes:
        if (node_id in full_graph.in_edges and
            len(full_graph.in_edges[node_id]) > 0):
            for in_node in full_graph.in_edges[node_id]:
                if in_node.stage_id != nodes[node_id].stage_id and check_stages:
                    # Skip hidden inputs.
                    if full_graph.nodes[in_node.node_id].node_desc.startswith("hidden"):
                        continue
                    input_names[in_node.node_id] = "input%d" % counter
                    counter += 1
        else:
            if graph.nodes[node_id].node_desc.startswith("Input"):
                input_names[node_id] = "input%d" % counter
                counter += 1
    return input_names

def get_output_names(graph, full_graph, counter):
    # Figure out the outputs of this sub-graph, which are the nodes in the
    # sub-graph with edges out of the sub-graph.
    nodes = graph.nodes
    output_names = {}
    for node_id in nodes:
        if (node_id in full_graph.edges and
            len(full_graph.edges[node_id]) > 0):
            for out_node in full_graph.edges[node_id]:
                if out_node.stage_id != nodes[node_id].stage_id:
                    if full_graph.nodes[node_id].node_desc.startswith("hidden"):
                        continue
                    output_names[node_id] = "out%d" % counter
                    counter += 1
        else:
            output_names[node_id] = "out%d" % counter
            counter += 1
    return output_names, counter

def convert_subgraph_to_module(graph, full_graph, num_subgraphs, module_name, initialize_weights,
                               model_template_filename, output_filename):
    model_template = open(model_template_filename, 'r').read()
    nodes = graph.topological_sort()
    import_statements = []
    module_methods = []

    counter = 0
    layer_names = {}
    layer_names_and_declarations = []
    function_definition = []
    input_names = get_input_names(graph, full_graph)
    num_inputs = len(input_names)
    output_names = input_names.copy()
    sources = graph.sources()

    # Now, generate expressions for each node.
    # Iterate through nodes in topological order, and add output_name mappings for
    # each expression. Use this output_name mapping when generating expressions
    # in the model's implementation file.
    # TODO: Make sure that nodes with multiple inputs have the inputs in the
    # right order (even though this probably does not matter in practice).
    for node_id in input_names:
        output_name = "out%d" % counter
        function_definition.append("%s = %s.clone()" % (output_name,
                                                        input_names[node_id]))
        output_names[node_id] = output_name
        counter += 1

    for node in nodes:
        layer_call = None
        layer_name = "self.layer%d" % counter
        output_name = "out%d" % counter
        layer_declaration = "torch.nn.%s" % (
            node.node_desc.replace("inplace", "inplace=True"))
        layer_names[node.node_id] = layer_name
        if node.node_id not in output_names:
            output_names[node.node_id] = output_name

        # Skip layers that don't need a declaration (example: '+=').
        for declaration in declaration_specialcase:
            if node.node_desc.startswith(declaration):
                found = True
                if declaration == "EmuBidirLSTM":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    layer_declaration = "EmuBidirLSTM(%d, %d)" % (input_size, hidden_size)
                    import_statements.append("from seq2seq.models.encoder import EmuBidirLSTM")
                elif declaration == "RecurrentAttention":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    context_size = int(m.group(1))
                    layer_declaration = "RecurrentAttention(%d, %d, %d)" % (input_size, hidden_size, context_size)
                    import_statements.append("from seq2seq.models.decoder import RecurrentAttention")
                elif declaration == "Classifier":
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    in_features = int(m.group(1))
                    out_features = int(m.group(2))
                    layer_declaration = "Classifier(%d, %d)" % (in_features, out_features)
                    import_statements.append("from seq2seq.models.decoder import Classifier")
                elif declaration == "MaskConv":
                    node_desc = node.node_desc
                    modules = node_desc.split("    ")[1:-1]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=True")
                        module_declarations.append(module_declaration)
                    layer_declaration = "MaskConv(torch.nn.Sequential(%s))" % ",\n            ".join(module_declarations)
                    import_statements.append("from model import MaskConv")
                    module_methods.append("""def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in %s.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()""" % layer_name)
                elif declaration == "BatchRNN":
                    if "batch_norm" in node.node_desc:
                        batch_norm = True
                    else:
                        batch_norm = False
                    if "LSTM" in node.node_desc:
                        rnn_type = "torch.nn.LSTM"
                        m = re.search(r'LSTM\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    elif "GRU" in node.node_desc:
                        rnn_type = "torch.nn.GRU"
                        m = re.search(r'GRU\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    else:
                        # TODO: Do something else?
                        pass
                    # TODO: Pass remaining arguments.
                    # TODO: Get hidden and input size.
                    layer_declaration = "BatchRNN(%d, %d, rnn_type=%s, batch_norm=%s, bidirectional=%s)" % (
                        input_size, hidden_size, rnn_type, batch_norm, bidirectional)
                    import_statements.append("from model import BatchRNN")
                elif declaration == "ResizeInput":
                    layer_declaration = "ResizeInput()"
                    import_statements.append("from model import ResizeInput") 
                elif declaration == "SequenceWise":
                    node_desc = node.node_desc
                    modules = node_desc[:-2].split("  ")[1:]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=True")
                        module_declarations.append(module_declaration)
                    layer_declaration = "SequenceWise(torch.nn.Sequential(%s))" % ",\n            ".join(module_declarations)
                    import_statements.append("from model import SequenceWise")
                elif declaration == "InferenceBatchSoftmax":
                    layer_declaration = "InferenceBatchSoftmax()"
                    import_statements.append("from model import InferenceBatchSoftmax")
                elif declaration == "EmbeddingBag": #TODO: other cases
                    layer_declaration = layer_declaration.replace("mode=sum", "mode=\"sum\", sparse=True")
                    #print(layer_declaration)
                break

        import_statements = list(set(import_statements))
        found = False
        for declaration in declaration_whitelist:
            if node.node_desc.startswith(declaration):
               found = True
        if not found:
            layer_names_and_declarations.append((layer_name, layer_declaration))

        if node.node_id in full_graph.in_edges:
            in_edges = full_graph.in_edges[node.node_id]
        else:
            in_edges = []
        if len(in_edges) == 0 and node.node_desc.startswith("Input"):
            pass  # Don't need to do anything for this case.
        else:
            if node.node_desc.startswith("Size"):
                assert(len(in_edges) == 1)
                m = re.search(r'Size\((-?\d+)\)', node.node_desc)
                idx = int(m.group(1))
                layer_call = "%s = %s.size(%d)" % (output_name,
                                                   output_names[in_edges[0].node_id],
                                                   idx)
            elif node.node_desc.startswith("View"):
                size_node_ids = []
                input_node_id = None
                for i in range(len(in_edges)):
                    if in_edges[i].node_desc.startswith("Size"):
                        size_node_id = in_edges[i].node_id
                        size_node_ids.append(size_node_id)
                    else:
                        input_node_id = in_edges[i].node_id
                m = re.search(r'View\((-?\d+)\)', node.node_desc)
                if m is None:
                    size_output_names = [output_names[size_node_id] for size_node_id in size_node_ids]
                    layer_call = "%s = %s.view(%s)" % (output_name,
                                                       output_names[input_node_id],
                                                       ", ".join(size_output_names))
                else:
                    size = int(m.group(1))
                    layer_call = "%s = %s.view(%s, %d)" % (output_name,
                                                           output_names[input_node_id],
                                                           output_names[size_node_id],
                                                           size)
            elif node.node_desc.startswith("__getitem__"):
                assert(len(in_edges) == 1)
                m = re.search(r'__getitem__\((.+)\)', node.node_desc)
                # TODO: a temporary solution
                # print(re.search(r'\d+', m.group(1)).group())
                idx = int(re.search(r'\d+', m.group(1)).group())
                if "hidden" in in_edges[0].node_desc:
                    layer_call = "%s = None" % output_name
                else:
                    layer_call = "%s = %s[%d]" % (output_name,
                                                  output_names[in_edges[0].node_id],
                                                  idx)
            elif node.node_desc.startswith("Add"):
                assert(len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s + %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Mul"):
                assert(len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s * %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Concat"):
                m = re.search(r'Concat\((-?\d+)\)', node.node_desc)
                dim = int(m.group(1))
                layer_call = "%s = torch.cat([%s], %d)" % (
                    output_name,
                    ", ".join([output_names[in_node.node_id]
                               for in_node in in_edges]), dim)
            elif node.node_desc.startswith("Bmm"): #TODO: a, b may need to swap
                assert(len(in_edges) == 2)
                layer_call = "%s = torch.bmm(%s)" % (
                    output_name,
                    ", ".join([output_names[in_node.node_id]
                               for in_node in in_edges]))
            elif node.node_desc.startswith("Transpose"):
                m = re.search(r'Transpose\((.+)\)', node.node_desc)
                args = m.group(1)
                assert(len(in_edges) == 1)
                node1 = in_edges[0]
                layer_call = "%s = %s.transpose(%s)" % (output_name, output_names[node1.node_id],
                                                        args)
            elif node.node_desc.startswith("hidden"):
                pass
            elif node.node_desc == "self.get_seq_lens":
                assert(len(in_edges) == 1)
                in_node = in_edges[0]
                layer_call = "%s = %s(%s)" % (output_name, node.node_desc, output_names[in_node.node_id])
            else:
                layer_call = "%s = %s(%s)" % (output_name, layer_name,
                                              ", ".join([output_names[in_node.node_id]
                                                         for in_node in in_edges]))
        if layer_call is not None:
            function_definition.append(layer_call)
        counter += 1

    # Ensure that outputs of a module are returned in the same order as
    # the original model implementation.
    # TODO: This might not work as intended for sub-graphs.
    full_graph.populate_depths()
    graph_output_names, _ = get_output_names(graph, full_graph, 0)
    for key in graph_output_names:
        graph_output_names[key] = output_names[key]
    output_names_list = get_tensor_names_list(graph_output_names)
    num_outputs = len(output_names_list)
    function_definition.append("return %s" %
        get_output_tuple_str(output_names_list))

    # Layer declarations are added to the constructor of the module.
    # Function definitions are added to the `forward()' method of the
    # module.
    layer_declarations_str = "\n        ".join([
        "%s = %s" % (x[0], x[1]) for x in layer_names_and_declarations])
    if initialize_weights:
        layer_declarations_str += "\n        self._initialize_weights()"
        module_methods.append("""def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)""")
    function_definition_str = "\n        ".join(function_definition)
    input_names_list = get_tensor_names_list(input_names)
    input_names = ", ".join(input_names_list)
    model = model_template % {"layer_declarations": layer_declarations_str,
                              "function_definition": function_definition_str,
                              "module_name": module_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": "\n\n".join(module_methods)}
    with open(output_filename, 'w') as f:
        f.write(model)
    return num_inputs, num_outputs

def fuse_subgraphs_to_module(graph, subgraphs, model_name, initialize_weights,
                             model_template_filename, output_filename):
    model_template = open(model_template_filename, 'r').read()

    # PyTorch modules are the names given to the generated stages (which are
    # of type torch.nn.Module).
    # Python modules are the names given to the filenames containing these
    # generated torch.nn.Modules.
    pytorch_modules = []
    python_modules = []
    for i in range(len(subgraphs)):
        pytorch_modules.append("Stage%d" % i)
        python_modules.append("stage%d" % i)

    layer_declarations = []
    function_definition = []
    for i, pytorch_module in enumerate(pytorch_modules):
        layer_declarations.append("self.stage%d = %s()" % (
            i, pytorch_module))
    if initialize_weights:
        layer_declarations.append("self._initialize_weights()")

    output_counter = 0
    output_names = {}
    graph_input_names = get_input_names(graph, graph, check_stages=False)
    for key in graph_input_names:
        output_names[key] = graph_input_names[key]
    subgraph_inputs = []
    subgraph_outputs = []
    for i, subgraph in enumerate(subgraphs):
        subgraph_input_names = get_input_names(subgraph, graph)
        subgraph_output_names, output_counter = get_output_names(
            subgraph, graph, output_counter)
        for key in subgraph_input_names:
            subgraph_input_names[key] = output_names[key]
        for key in subgraph_output_names:
            output_names[key] = subgraph_output_names[key]

        function_definition.append("%s = self.stage%d(%s)" % (
            get_output_tuple_str(get_tensor_names_list(subgraph_output_names)),
            i, ", ".join(get_tensor_names_list(subgraph_input_names))))
        subgraph_inputs.append(get_tensor_names_list(subgraph_input_names))
        subgraph_outputs.append(get_tensor_names_list(subgraph_output_names))

    function_definition.append("return %s" %
        get_output_tuple_str(get_tensor_names_list(subgraph_output_names)))
    function_definition_str = "\n        ".join(function_definition)

    import_statements = ["from .%s import %s" % (python_module, pytorch_module)
                         for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)]
    input_names = get_input_names(graph, graph, check_stages=False)
    input_names = ", ".join(get_tensor_names_list(input_names))
    model = model_template % {"layer_declarations": "\n        ".join(layer_declarations),
                              "function_definition": function_definition_str,
                              "module_name": model_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": ""}  # TODO: Figure out if we need to pass in other module_methods here?

    print("Done with sub-graph fusion...")

    with open(output_filename, 'w') as f:
        f.write(model)
    return python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert profile graphs to generated model description")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Input profile filename")
    parser.add_argument("--model_template_filename", default="templates/model.py.template",
                        help="Model template filename")
    parser.add_argument("--init_template_filename", default="templates/__init__.py.template",
                        help="__init__.py template filename")
    parser.add_argument("--conf_template_filename", default="templates/conf.json.template",
                        help="Conf template filename")
    parser.add_argument("--stage_to_num_ranks_map", type=str, default=None,
                        help="Stage split")
    parser.add_argument('-n', "--model_name", required=True,
                        help="Name of model class")
    parser.add_argument('-a', "--arch", required=True,
                        help="Human-readable architecture name")
    parser.add_argument('-o', "--output_directory", required=True,
                        help="Full path of output model directory")
    args = parser.parse_args()

    # mkdir output_directory.
    subprocess.check_output("mkdir -p %s" % args.output_directory, shell=True)

    input_node = graph.Node("input_node", node_desc="Input")
    full_graph = graph.Graph.from_str(open(args.profile_filename, 'r').read())
    initialize_weights = (args.arch == "vgg16" or args.arch == "resnet50")
    input_node.stage_id = 0
    sinks = full_graph.sinks()
    # Remove all unneeded sinks that are not used, makes code generation easier.
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            full_graph.remove_node(sink)
    subgraphs = full_graph.partition_graph()

    for i, subgraph in enumerate(subgraphs):
        module_name = "Stage%d" % i
        module_filename = "stage%d.py" % i
        if len(subgraphs) == 1:
            module_name = args.model_name
            module_filename = "%s.py" % args.arch
        num_inputs, num_outputs = convert_subgraph_to_module(subgraph, full_graph, len(subgraphs),
                                                             module_name, initialize_weights,
                                                             args.model_template_filename,
                                                             os.path.join(args.output_directory,
                                                                          module_filename))
        print("Done generating %s..." % module_filename)

    model = []
    import_statements = ["from .%s import %s" % (args.arch, args.model_name)]
    pytorch_modules = None
    if len(subgraphs) > 1:
        python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs = \
            fuse_subgraphs_to_module(full_graph, subgraphs, args.model_name,
                                     initialize_weights,
                                     args.model_template_filename,
                                     os.path.join(args.output_directory,
                                                  "%s.py" % args.arch))
        model = ["(%s(), [%s], [%s])" % (x[0],
                                         ", ".join(["\"%s\"" % y for y in x[1]]),
                                         ", ".join(["\"%s\"" % y for y in x[2]]))
                 for x in zip(pytorch_modules, subgraph_inputs,
                              subgraph_outputs)]
        model.append("(criterion, [\"%s\"], [\"loss\"])" % subgraph_outputs[-1][0])
        import_statements.extend(
            ["from .%s import %s" % (python_module, pytorch_module)
             for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)])
    else:
        inputs = ["\"input%d\"" % i for i in range(num_inputs)]
        assert(num_outputs == 1)
        model.append("(%s.%s(), [%s], [\"output\"])" % (args.arch, args.model_name, ", ".join(inputs)))
        model.append("(criterion, [\"output\"], [\"loss\"])")

    with open(os.path.join(args.output_directory, "__init__.py"), 'w') as f1, \
         open(args.init_template_filename, 'r') as f2:
        template = f2.read()
        init = template % {
            "arch": args.arch,
            "import_statements": "\n".join(import_statements),
            "model": ",\n        ".join(model),
            "full_model": "%s()" % args.model_name
        }
        f1.write(init)

    if args.stage_to_num_ranks_map is not None:
        stage_to_num_ranks_map = args.stage_to_num_ranks_map.split(",")
        stage_to_num_ranks_map = [(int(x.split(":")[0]), int(x.split(":")[1]))
                      for x in stage_to_num_ranks_map]
        num_stages = 0
        for (stage_id, replication_factor) in stage_to_num_ranks_map:
            num_stages += replication_factor
        print(len(stage_to_num_ranks_map), len(python_modules))
        assert(len(stage_to_num_ranks_map) == len(pytorch_modules))
        num_modules = len(pytorch_modules) + 1  # Add 1 for criterion.
    elif pytorch_modules is None:
        num_stages = 1
        num_modules = 2  # Add 1 for criterion.
    else:
        num_stages = len(pytorch_modules)
        num_modules = len(pytorch_modules) + 1  # Add 1 for criterion.
    all_template_args = []
    all_template_args.append({
        "module_to_stage_map": [0] * num_modules,
        "stage_to_rank_map": str({"0": list(range(num_stages))}).replace("'", "\"")
    })
    all_template_args.append({
        "module_to_stage_map": list(range(num_modules-1)) + [num_modules-2],
        "stage_to_rank_map": str({str(i): [i] for i in range(num_modules-1)}).replace("'", "\"")
    })
    if args.stage_to_num_ranks_map is not None:
        stage_to_rank_map = {}
        ranks_so_far = 0
        for i in range(num_modules-1):
            stage_to_rank_map[str(i)] = list(range(ranks_so_far,
                                                   ranks_so_far + stage_to_num_ranks_map[i][1]))
            ranks_so_far += stage_to_num_ranks_map[i][1]
        stage_to_rank_map = str(stage_to_rank_map).replace("'", "\"")
        all_template_args.append({
            "module_to_stage_map": list(range(num_modules-1)) + [num_modules-2],
            "stage_to_rank_map": stage_to_rank_map
        })
    for conf_filename, template_args in zip(
        ["dp_conf.json", "mp_conf.json", "hybrid_conf.json"], all_template_args):
        with open(os.path.join(args.output_directory, conf_filename), 'w') as f1, \
             open(args.conf_template_filename, 'r') as f2:
            template = f2.read()
            conf = template % template_args
            f1.write(conf)

