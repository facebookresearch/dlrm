# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
sys.path.append("..")
sys.path.append("../..")
import graph

def convert_graph(graph_filename, output_directory, arch):
    print("Processing %s..." % graph_filename)
    gr = graph.Graph.from_str(open(graph_filename, 'r').read())
    compressed_gr = gr.compress_branches()
    gr.check_fidelity(compressed_gr)
    output_directory = os.path.join(output_directory, arch)
    compressed_gr.to_dot(os.path.join(output_directory, "graph"))
    with open(os.path.join(output_directory, "graph.txt"), 'w') as f:
        f.write(str(compressed_gr))


if __name__ == '__main__':
    output_directory = "compressed_graphs"
    convert_graph("../baselines/profiles/inception_v3/graph.txt", output_directory, 'inception_v3')
    convert_graph("../baselines/profiles/nasnetamobile/graph.txt", output_directory, 'nasnetamobile')
    convert_graph("../baselines/profiles/nasnetalarge/graph.txt", output_directory, 'nasnetalarge')
