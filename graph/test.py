# from PipeDream

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graph

def get_graph_from_filename(graph_filename):
    with open(graph_filename, 'r') as f:
        graph_str = f.read()
    gr = graph.Graph.from_str(graph_str)
    return gr

def test_topological_sort(graph_filename):
    gr = get_graph_from_filename(graph_filename)
    # TODO: Assert that this topological ordering is correct.
    # print(gr.topological_sort())

def test_predecessors(graph_filename, node):
    gr = get_graph_from_filename(graph_filename)
    print("test_predecessors:", node, [node.node_id for node in gr.predecessors(node)])

def test_augment_antichain(graph_filename, antichain):
    gr = get_graph_from_filename(graph_filename)
    print("test_augment_antichain:", antichain,
        gr.augment_antichain(antichain))

def test_deaugment_augmented_antichain(graph_filename, augmented_antichain):
    gr = get_graph_from_filename(graph_filename)
    print("test_deaugment_augmented_antichain:", augmented_antichain,
        gr.deaugment_augmented_antichain(augmented_antichain))

def test_next_antichains(graph_filename, antichain):
    gr = get_graph_from_filename(graph_filename)
    next_antichains = gr.next_antichains(antichain)
    print("test_next_antichains:", antichain, "True")

def test_antichain_dag(graph_filename):
    gr = get_graph_from_filename(graph_filename)
    print("test_antichain_dag:",
        [node.antichain for node in gr.antichain_dag().topological_sort()])

def test_is_series_parallel(graph_filename, arch):
    gr = get_graph_from_filename(graph_filename)
    print("is_series_parallel (%s):" % arch, gr.is_series_parallel(arch))

def test_check_isomorphism(graph1_filename, graph2_filename, arch):
    gr1 = get_graph_from_filename(graph1_filename)
    gr2 = get_graph_from_filename(graph2_filename)
    gr1.check_isomorphism(gr2)
    print("check_isomorphism (%s):" % arch, "True")

def test_partitioning(graph_filename, model):
    gr = get_graph_from_filename(graph_filename)
    sub_graphs = gr.partition_graph()
    assert(len(sub_graphs) == 3)
    print("test_partitioning (%s): True" % model)


if __name__ == '__main__':
    test_topological_sort("test_graphs/test.txt")
    test_topological_sort("../baselines/profiles/resnet18/graph.txt")
    test_topological_sort("../baselines/profiles/resnet50/graph.txt")

    test_predecessors("test_graphs/test.txt", "0")
    test_predecessors("test_graphs/test.txt", "5")
    test_predecessors("test_graphs/test2.txt", "3")
    test_predecessors("test_graphs/test2.txt", "2")

    test_augment_antichain("test_graphs/test2.txt", ["2"])
    test_augment_antichain("test_graphs/test2.txt", ["3"])

    test_deaugment_augmented_antichain("test_graphs/test2.txt", ["1", "3"])
    test_deaugment_augmented_antichain("test_graphs/test2.txt", ["3"])
    test_deaugment_augmented_antichain("test_graphs/test2.txt", ["0", "3"])

    test_next_antichains("test_graphs/test2.txt", ["2"])
    test_next_antichains("test_graphs/test2.txt", ["3"])
    test_next_antichains("test_graphs/test2.txt", ["0"])

    test_antichain_dag("test_graphs/test.txt")
    test_antichain_dag("test_graphs/test2.txt")
    test_antichain_dag("../baselines/profiles/resnet18/graph.txt")

    test_is_series_parallel("../baselines/profiles/resnet18/graph.txt", "resnet18")
    test_is_series_parallel("../baselines/profiles/resnet50/graph.txt", "resnet50")
    test_is_series_parallel("../baselines/profiles/inception_v3/graph.txt", "inception_v3")
    test_is_series_parallel("../baselines/profiles/nasnetamobile/graph.txt", "nasnetamobile")

    test_check_isomorphism("../baselines/profiles/resnext50/graph.txt",
                           "test_graphs/resnext50_generated.txt", "resnext50")

    test_partitioning("test_graphs/vgg16_partitioned.txt", "vgg16")
