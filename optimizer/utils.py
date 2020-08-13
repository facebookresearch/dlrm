# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os

import sys
sys.path.append("..")
import graph
try: # a quick fix
    # Python 2
    xrange
except NameError:
    # Python 3
    xrange = range

def parse_profile_file(profile_filename):
    with open(profile_filename, 'r') as f:
        csv_reader = csv.reader(f)
        line_id = 0
        profile_data = []
        for line in csv_reader:
            if line_id == 0:
                header = line
                num_minibatches = None
                for header_elem in header:
                    if "Forward pass time" in header_elem:
                        num_minibatches = int(header_elem.split("(")[1].rstrip(")"))
            else:
                total_time = float(line[header.index("Total time")]) / num_minibatches
                for i in xrange(len(header)):
                    if "Output Size" in header[i]:
                        if line[i] == '':
                            output_size = 0
                        else:
                            output_size = float(line[i].replace(",", ""))
                        break
                parameter_size = float(line[header.index("Parameter Size (floats)")].replace(",", ""))
                profile_data.append((total_time, output_size * 4.0, parameter_size * 4.0))
            line_id += 1
    return profile_data

def parse_profile_file_to_graph(profile_filename, directory):
    gr = graph.Graph()
    node_id = 0
    with open(profile_filename, 'r') as f:
        csv_reader = csv.reader(f)
        line_id = 0
        profile_data = []
        prev_node = None
        for line in csv_reader:
            if line_id == 0:
                header = line
                num_minibatches = None
                for header_elem in header:
                    if "Forward pass time" in header_elem:
                        num_minibatches = int(header_elem.split("(")[1].rstrip(")"))
            else:
                total_time = float(line[header.index("Total time")]) / num_minibatches
                for i in xrange(len(header)):
                    if "Output Size" in header[i]:
                        if line[i] == '':
                            output_size = 0
                        else:
                            output_size = float(line[i].replace(",", ""))
                        break
                parameter_size = float(line[header.index("Parameter Size (floats)")].replace(",", ""))
                node_desc = line[header.index("Layer Type")]
                node = graph.Node("node%d" % node_id, node_desc=node_desc,
                                  compute_time=total_time * 1000,
                                  parameter_size=(4.0 * parameter_size),
                                  activation_size=(output_size * 4.0))
                node_id += 1
                if prev_node is not None:
                    gr.add_edge(prev_node, node)
                prev_node = node
            line_id += 1

    gr.to_dot(os.path.join(directory, "graph.dot"))
    with open(os.path.join(directory, "graph.txt"), 'w') as f:
        f.write(str(gr))

    return gr
