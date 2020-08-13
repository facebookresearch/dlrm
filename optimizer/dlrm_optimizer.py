# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Description: Performance Modeling and Optimization for PipeDLRM.
# Author: Yanzhao Wu (yanzhaowu@fb.com)

import sys
sys.path.append("..")

import graph
import argparse

# constant variables

netBandwidth = 1000000000.0 # 1GB/s


def get_stage_statistics(gr):
    nodes = gr.nodes
    stage_nodes_map = dict()
    stage_time = dict()
    serial_time = 0.0
    for n in nodes.values():
        stage_id = n.stage_id
        #print(n)
        stage_nodes = stage_nodes_map.get(stage_id, list())
        stage_nodes.append(n.node_id)
        stage_nodes_map[stage_id] = stage_nodes

        cum_forward_time, cum_backward_time = stage_time.get(stage_id, [0, 0])
        cum_forward_time += n.forward_compute_time
        cum_backward_time += n.backward_compute_time
        stage_time[stage_id] = [cum_forward_time, cum_backward_time]

        serial_time += (n.forward_compute_time + n.backward_compute_time)
    #print(stage_nodes_map)

    for cur_stage_id, stage_node_keys in stage_nodes_map.items():
        cum_activation_size = 0
        pre_nodes = set() # avoid duplications
        next_nodes = set()
        if cur_stage_id > 0: # previous nodes
            for cur_node_key in stage_node_keys:
                for pre_node in gr.in_edges[cur_node_key]:
                    #print(pre_node)
                    if pre_node.stage_id == cur_stage_id-1:
                        pre_nodes.add(pre_node)
            #print(len(pre_nodes))
            for pre_node in pre_nodes:
                cum_activation_size += pre_node.activation_size

        if cur_stage_id < len(stage_nodes_map)-1: # successors
            for cur_node_key in stage_node_keys:
                for next_node in gr.edges[cur_node_key]:
                    #print(next_node.node_id)
                    if next_node.stage_id == cur_stage_id+1:
                        next_nodes.add(gr.nodes[cur_node_key]) # add nodes in this stage
            #print(len(next_nodes))
            for next_node in next_nodes:
                cum_activation_size += next_node.activation_size
        #print(cur_stage_id, len(pre_nodes), len(next_nodes))
        stage_time[cur_stage_id].append(cum_activation_size*1000.0/netBandwidth)
        #print(cum_activation_size)

    #print(stage_time)
    max_forward_time = 0
    max_backward_time = 0
    for stage_id, (fwd_t, bwd_t, comm_t) in stage_time.items():
        max_forward_time = max(max_forward_time, fwd_t + comm_t)
        max_backward_time = max(max_backward_time, bwd_t + comm_t)

    stage_statistics_list = list()
    for item in stage_time.items():
        stage_statistics_list.append([item[0], item[1][0], item[1][1], item[1][2]])

    stage_statistics_list.sort()

    result = "without pipeline: {:.2f}, ideal pipeline speedup: {:.2f}x\n".format(serial_time, 1.0 * serial_time/(max_forward_time+max_backward_time))
    result += "#stages: " + str(len(stage_statistics_list)) + "\n"
    for ss in stage_statistics_list:
        result += "stage " + str(ss[0])
        result += ' forward time: {:.2f} backward time: {:.2f} communication time: {:.2f}\n'.format(ss[1], ss[2], ss[3])

    return result

# just for test
if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Profile filename")

    args = parser.parse_args()
    gr = graph.Graph().from_str(open(args.profile_filename, "r").read())
    print(get_stage_statistics(gr))
